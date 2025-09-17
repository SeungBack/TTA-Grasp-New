
import os
import numpy as np
import open3d as o3d
import torch
import glob
import h5py

from torch.utils.data import Dataset
from dataset.data_utils import *
# from data_utils import *
from tqdm import tqdm
import multiprocessing



class GraspEvalDataset(Dataset):
    def __init__(self, g1b_root, acronym_root, split='train', return_score=True, use_normal=False):
        # load acronym
        self.split = split
        if split == 'train':
            syn_h5_paths = glob.glob(acronym_root + '/*/*.h5')
            print('Total number of syn h5 files for {}:'.format(split), len(syn_h5_paths))
            syn_h5_paths = []
        else:
            syn_h5_paths = []
        # load g1b
        real_h5_paths = []
        for camera in ['kinect', 'realsense']:
            real_h5_paths += sorted(glob.glob(os.path.join(g1b_root, camera, split) + '/*.h5'))
            # if split != 'train':
                # real_h5_paths = real_h5_paths[:5000]
        print('Total number of real h5 files for {}:'.format(split), len(real_h5_paths))
        self.h5_paths = syn_h5_paths + real_h5_paths
        
        self.return_score = return_score
        self.use_normal = use_normal

    def __len__(self):
        return len(self.h5_paths)

    def __getitem__(self, index):
        
        with h5py.File(self.h5_paths[index], 'r') as f:
            obj_cloud = f['obj_cloud'][()]
            gripper_cloud = f['gripper_cloud'][()] 
            score = f['score'][()] 
            
        if self.split == 'train':
            obj_cloud = corrupt_dropout_global(obj_cloud, drop_rates=[0.01, 0.2])
            obj_cloud = corrupt_dropout_local(obj_cloud, npoints=[10, 200])
            obj_cloud = delete_random_knn_regions(obj_cloud, 10, 10)
            obj_cloud = corrupt_add_global(obj_cloud, npoints=[10, 100])
            obj_cloud = corrupt_add_local(obj_cloud, npoints=[10, 100])
            obj_cloud = to_fixed_size_pointcloud(obj_cloud, 1024)
            obj_cloud = corrupt_jitter(obj_cloud, sigmas = [0.00, 0.003])
            obj_cloud = scale_and_translate(obj_cloud, scale=[0.97, 1.03], translate=[-0.005, 0.005])
            obj_cloud = rotate(obj_cloud, angle=[-np.pi/60, np.pi/60])
            # visualize = False
            
            # visualize 
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(obj_cloud)
            # pcd.paint_uniform_color([0.5, 0.5, 0.5])
            

            # # pcd_ = o3d.geometry.PointCloud()
            # # pcd_.points = o3d.utility.Vector3dVector(obj_cloud)
            # # pcd_.paint_uniform_color([0.0, 0.5, 1.0])
            
            # gripper_pcd = o3d.geometry.PointCloud()
            # gripper_pcd.points = o3d.utility.Vector3dVector(gripper_cloud)
            # gripper_pcd.paint_uniform_color([1, 0, 0])
            # o3d.visualization.draw_geometries([pcd, gripper_pcd])
            
            # if self.use_normal:
            #     pcd = o3d.geometry.PointCloud()
            #     pcd.points = o3d.utility.Vector3dVector(obj_cloud)
            #     pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=10))
            #     pcd.orient_normals_consistent_tangent_plane(10)
            #     normals = np.asarray(pcd.normals)
            #     obj_cloud = np.concatenate((obj_cloud, normals), axis=1)
                
          
            
        obj_cloud = np.array(obj_cloud, dtype=np.float32)
        gripper_cloud = np.array(gripper_cloud, dtype=np.float32)
        score = np.round(score, 2)    

        # score >= 0.7: good, 0.3 <= score < 0.7: mid, score < 0.3: bad
        if self.return_score:
            # if score > 0.9: score = 0.9
            # if score > 0.9: score = 1.0
            return obj_cloud, gripper_cloud, score
        else:
            if score >= 0.3:
                category = 1
            else:
                category = 0
            return obj_cloud, gripper_cloud, category


import torch
from torch.utils.data import Sampler, Dataset
import numpy as np
from typing import List, Iterator


class BalancedBatchSampler(Sampler):
    """
    BatchSampler that ensures a fixed amount of real and synthetic samples per batch.
    
    Args:
        dataset: The dataset to sample from
        batch_size: Size of mini-batch
        real_ratio: The ratio of real samples in each batch (0.0 to 1.0)
        drop_last: If True, the sampler will drop the last batch if its size would be less than batch_size
    """
    def __init__(self, dataset: GraspEvalDataset, batch_size: int, real_ratio: float = 0.5, drop_last: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.real_ratio = max(0.0, min(1.0, real_ratio))  # Ensure real_ratio is between 0 and 1
        
        # Split indices into real and synthetic
        self.real_indices = []
        self.syn_indices = []
        
        # Identify real and synthetic samples based on file paths
        for i, path in enumerate(dataset.h5_paths):
            if 'acronym' in path.lower():
                self.syn_indices.append(i)
            else:
                self.real_indices.append(i)
                
        self.real_indices = np.array(self.real_indices)
        self.syn_indices = np.array(self.syn_indices)
        
        # Calculate number of real and synthetic samples per batch
        self.real_samples_per_batch = int(self.batch_size * self.real_ratio)
        self.syn_samples_per_batch = self.batch_size - self.real_samples_per_batch
        
        # Calculate number of batches
        self.num_real_batches = len(self.real_indices) // self.real_samples_per_batch
        self.num_syn_batches = len(self.syn_indices) // self.syn_samples_per_batch
        self.num_batches = min(self.num_real_batches, self.num_syn_batches)
        
        if not drop_last and (len(self.real_indices) % self.real_samples_per_batch != 0 or 
                              len(self.syn_indices) % self.syn_samples_per_batch != 0):
            self.num_batches += 1
    
    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle indices for each epoch
        real_indices = self.real_indices.copy()
        syn_indices = self.syn_indices.copy()
        np.random.shuffle(real_indices)
        np.random.shuffle(syn_indices)
        
        real_idx, syn_idx = 0, 0
        for _ in range(self.num_batches):
            batch_indices = []
            
            # Add real samples
            if real_idx + self.real_samples_per_batch <= len(real_indices):
                batch_indices.extend(real_indices[real_idx:real_idx + self.real_samples_per_batch])
                real_idx += self.real_samples_per_batch
            elif not self.drop_last:
                # Add remaining real samples
                batch_indices.extend(real_indices[real_idx:])
                real_idx = len(real_indices)
            
            # Add synthetic samples
            if syn_idx + self.syn_samples_per_batch <= len(syn_indices):
                batch_indices.extend(syn_indices[syn_idx:syn_idx + self.syn_samples_per_batch])
                syn_idx += self.syn_samples_per_batch
            elif not self.drop_last:
                # Add remaining synthetic samples
                batch_indices.extend(syn_indices[syn_idx:])
                syn_idx = len(syn_indices)
            
            # Shuffle the batch indices to avoid having all real samples followed by all synthetic samples
            np.random.shuffle(batch_indices)
            yield batch_indices
    
    def __len__(self) -> int:
        return self.num_batches


# Example usage:
def create_balanced_dataloader(dataset, batch_size, real_ratio=0.25, num_workers=4, drop_last=True):
    """
    Create a DataLoader with balanced sampling between real and synthetic data.
    
    Args:
        dataset: GraspEvalDataset instance
        batch_size: Size of each batch
        real_ratio: Ratio of real samples in each batch (0.0 to 1.0)
        num_workers: Number of worker processes for data loading
        drop_last: Whether to drop the last incomplete batch
    
    Returns:
        DataLoader with balanced sampling
    """
    sampler = BalancedBatchSampler(dataset, batch_size, real_ratio, drop_last)
    return torch.utils.data.DataLoader(
        dataset, 
        batch_sampler=sampler,
        num_workers=num_workers,
        # multiprocessing_context=multiprocessing.get_context("spawn")
    )

import torch
from torch.utils.data import Sampler, Dataset
import numpy as np
from typing import List, Iterator
import h5py


class GraspQualityBatchSampler(Sampler):
    """
    BatchSampler that ensures a fixed ratio of good and bad grasp samples per batch.
    
    Args:
        dataset: The GraspEvalDataset to sample from
        batch_size: Size of mini-batch
        good_ratio: The ratio of good grasp samples in each batch (0.0 to 1.0)
        threshold: Score threshold to classify grasps as good (>=threshold) or bad (<threshold)
        drop_last: If True, the sampler will drop the last batch if its size would be less than batch_size
    """
    def __init__(self, dataset, batch_size: int, good_ratio: float = 0.5, threshold: float = 0.3, drop_last: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.good_ratio = max(0.0, min(1.0, good_ratio))  # Ensure good_ratio is between 0 and 1
        self.threshold = threshold
        
        good_path = 'good_grasps.npy'
        bad_path = 'bad_grasps.npy'
        # if os.path.exists(good_path) and os.path.exists(bad_path):
        #     self.good_indices = np.load(good_path)
        #     self.bad_indices = np.load(bad_path)
        #     print('Loaded good and bad grasp indices from files.')
        # else:
            
        # Split indices into good and bad grasps
        self.good_indices = []
        self.bad_indices = []
        
        # Scan through dataset to identify good and bad grasps based on scores
        print("Preprocessing dataset to identify good and bad grasps...")
        for i in tqdm(range(len(dataset))):
            # Read score directly from the H5 file without loading the full data
            with h5py.File(dataset.h5_paths[i], 'r') as f:
                score = f['score'][()]
            
            if score >= threshold:
                self.good_indices.append(i)
            else:
                self.bad_indices.append(i)
                
        self.good_indices = np.array(self.good_indices)
        self.bad_indices = np.array(self.bad_indices)
            # Save the indices to files for future use
            # np.save(good_path, self.good_indices)
            # np.save(bad_path, self.bad_indices)
        print('Saved good and bad grasp indices to files.')
        
        print(f"Found {len(self.good_indices)} good grasps and {len(self.bad_indices)} bad grasps")
        
        # Calculate number of good and bad samples per batch
        self.good_samples_per_batch = int(self.batch_size * self.good_ratio)
        self.bad_samples_per_batch = self.batch_size - self.good_samples_per_batch
        
        # Calculate number of batches
        self.num_good_batches = len(self.good_indices) // self.good_samples_per_batch
        self.num_bad_batches = len(self.bad_indices) // self.bad_samples_per_batch
        self.num_batches = min(self.num_good_batches, self.num_bad_batches)
        
        if not drop_last and (len(self.good_indices) % self.good_samples_per_batch != 0 or 
                              len(self.bad_indices) % self.bad_samples_per_batch != 0):
            self.num_batches += 1
    
    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle indices for each epoch
        good_indices = self.good_indices.copy()
        bad_indices = self.bad_indices.copy()
        np.random.shuffle(good_indices)
        np.random.shuffle(bad_indices)
        
        good_idx, bad_idx = 0, 0
        for _ in range(self.num_batches):
            batch_indices = []
            
            # Add good grasp samples
            if good_idx + self.good_samples_per_batch <= len(good_indices):
                batch_indices.extend(good_indices[good_idx:good_idx + self.good_samples_per_batch])
                good_idx += self.good_samples_per_batch
            elif not self.drop_last:
                # Add remaining good grasp samples
                batch_indices.extend(good_indices[good_idx:])
                good_idx = len(good_indices)
            
            # Add bad grasp samples
            if bad_idx + self.bad_samples_per_batch <= len(bad_indices):
                batch_indices.extend(bad_indices[bad_idx:bad_idx + self.bad_samples_per_batch])
                bad_idx += self.bad_samples_per_batch
            elif not self.drop_last:
                # Add remaining bad grasp samples
                batch_indices.extend(bad_indices[bad_idx:])
                bad_idx = len(bad_indices)
            
            # Shuffle the batch indices to mix good and bad samples
            np.random.shuffle(batch_indices)
            yield batch_indices
    
    def __len__(self) -> int:
        return self.num_batches


# Helper function to create a dataloader with balanced grasp quality
def create_balanced_grasp_quality_dataloader(dataset, batch_size, good_ratio=0.5, threshold=0.3, num_workers=4, drop_last=True):
    """
    Create a DataLoader with balanced sampling between good and bad grasps.
    
    Args:
        dataset: GraspEvalDataset instance
        batch_size: Size of each batch
        good_ratio: Ratio of good grasps in each batch (0.0 to 1.0)
        threshold: Score threshold to classify grasps as good (>=threshold) or bad (<threshold)
        num_workers: Number of worker processes for data loading
        drop_last: Whether to drop the last incomplete batch
    
    Returns:
        DataLoader with balanced grasp quality sampling
    """
    sampler = GraspQualityBatchSampler(dataset, batch_size, good_ratio, threshold, drop_last)
    return torch.utils.data.DataLoader(
        dataset, 
        batch_sampler=sampler,
        num_workers=num_workers,
        # Use spawn context if needed for multiprocessing
        # multiprocessing_context=multiprocessing.get_context("spawn")
    )


class CombinedBatchSampler(Sampler):
    """
    BatchSampler that ensures balanced sampling for both:
    1. Real vs synthetic data
    2. Good vs bad grasp quality
    
    Args:
        dataset: The GraspEvalDataset to sample from
        batch_size: Size of mini-batch
        real_ratio: The ratio of real samples in each batch (0.0 to 1.0)
        good_ratio: The ratio of good grasp samples in each batch (0.0 to 1.0)
        threshold: Score threshold to classify grasps as good (>=threshold) or bad (<threshold)
        cache_dir: Directory to cache indices for faster loading
        drop_last: If True, drop the last batch if smaller than batch_size
    """
    def __init__(self, dataset, batch_size: int, real_ratio: float = 0.5, good_ratio: float = 0.5, 
                 threshold: float = 0.3, cache_dir: str = './', drop_last: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.real_ratio = max(0.0, min(1.0, real_ratio))
        self.good_ratio = max(0.0, min(1.0, good_ratio))
        self.threshold = threshold
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # File paths for cached indices
        real_good_path = os.path.join(cache_dir, 'real_good_indices.npy')
        real_bad_path = os.path.join(cache_dir, 'real_bad_indices.npy')
        syn_good_path = os.path.join(cache_dir, 'syn_good_indices.npy')
        syn_bad_path = os.path.join(cache_dir, 'syn_bad_indices.npy')
        
        # Check if cached indices exist
        if (os.path.exists(real_good_path) and os.path.exists(real_bad_path) and
            os.path.exists(syn_good_path) and os.path.exists(syn_bad_path)):
            # Load cached indices
            self.real_good_indices = np.load(real_good_path)
            self.real_bad_indices = np.load(real_bad_path)
            self.syn_good_indices = np.load(syn_good_path)
            self.syn_bad_indices = np.load(syn_bad_path)
            print('Loaded cached indices from files.')
        else:
            # Initialize the four categories of indices
            self.real_good_indices = []
            self.real_bad_indices = []
            self.syn_good_indices = []
            self.syn_bad_indices = []
            
            # Scan through dataset to categorize all samples
            print("Preprocessing dataset to categorize samples...")
            for i in tqdm(range(len(dataset))):
                path = dataset.h5_paths[i]
                
                # Determine if real or synthetic
                is_real = 'acronym' not in path.lower()
                
                # Read score directly from the H5 file
                with h5py.File(path, 'r') as f:
                    score = f['score'][()]
                
                # Categorize based on both criteria
                if is_real:
                    if score >= threshold:
                        self.real_good_indices.append(i)
                    else:
                        self.real_bad_indices.append(i)
                else:
                    if score >= threshold:
                        self.syn_good_indices.append(i)
                    else:
                        self.syn_bad_indices.append(i)
            
            # Convert lists to numpy arrays
            self.real_good_indices = np.array(self.real_good_indices)
            self.real_bad_indices = np.array(self.real_bad_indices)
            self.syn_good_indices = np.array(self.syn_good_indices)
            self.syn_bad_indices = np.array(self.syn_bad_indices)
            
            # Save indices to cache
            np.save(real_good_path, self.real_good_indices)
            np.save(real_bad_path, self.real_bad_indices)
            np.save(syn_good_path, self.syn_good_indices)
            np.save(syn_bad_path, self.syn_bad_indices)
            print('Saved indices to cache.')
        
        print(f"Distribution of samples:")
        print(f"- Real good: {len(self.real_good_indices)}")
        print(f"- Real bad: {len(self.real_bad_indices)}")
        print(f"- Synthetic good: {len(self.syn_good_indices)}")
        print(f"- Synthetic bad: {len(self.syn_bad_indices)}")
        
        # Calculate samples per category per batch
        real_samples = int(batch_size * real_ratio)
        syn_samples = batch_size - real_samples
        
        self.real_good_samples = int(real_samples * good_ratio)
        self.real_bad_samples = real_samples - self.real_good_samples
        self.syn_good_samples = int(syn_samples * good_ratio)
        self.syn_bad_samples = syn_samples - self.syn_good_samples
        
        print(f"Samples per batch:")
        print(f"- Real good: {self.real_good_samples}")
        print(f"- Real bad: {self.real_bad_samples}")
        print(f"- Synthetic good: {self.syn_good_samples}")
        print(f"- Synthetic bad: {self.syn_bad_samples}")
        
        # Calculate number of complete batches
        self.num_real_good_batches = len(self.real_good_indices) // self.real_good_samples if self.real_good_samples > 0 else float('inf')
        self.num_real_bad_batches = len(self.real_bad_indices) // self.real_bad_samples if self.real_bad_samples > 0 else float('inf')
        self.num_syn_good_batches = len(self.syn_good_indices) // self.syn_good_samples if self.syn_good_samples > 0 else float('inf')
        self.num_syn_bad_batches = len(self.syn_bad_indices) // self.syn_bad_samples if self.syn_bad_samples > 0 else float('inf')
        
        # Total number of batches is limited by the smallest category
        self.num_batches = int(min(
            self.num_real_good_batches,
            self.num_real_bad_batches,
            self.num_syn_good_batches,
            self.num_syn_bad_batches
        ))
        
        if self.num_batches == 0:
            raise ValueError("Insufficient samples to create even one batch with the requested distribution.")
        
        print(f"Can create {self.num_batches} complete batches")
    
    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle all indices for each epoch
        real_good_indices = self.real_good_indices.copy()
        real_bad_indices = self.real_bad_indices.copy()
        syn_good_indices = self.syn_good_indices.copy()
        syn_bad_indices = self.syn_bad_indices.copy()
        
        np.random.shuffle(real_good_indices)
        np.random.shuffle(real_bad_indices)
        np.random.shuffle(syn_good_indices)
        np.random.shuffle(syn_bad_indices)
        
        # Track current index for each category
        real_good_idx, real_bad_idx = 0, 0
        syn_good_idx, syn_bad_idx = 0, 0
        
        for _ in range(self.num_batches):
            batch_indices = []
            
            # Add samples from each category
            if self.real_good_samples > 0:
                batch_indices.extend(real_good_indices[real_good_idx:real_good_idx + self.real_good_samples])
                real_good_idx += self.real_good_samples
            
            if self.real_bad_samples > 0:
                batch_indices.extend(real_bad_indices[real_bad_idx:real_bad_idx + self.real_bad_samples])
                real_bad_idx += self.real_bad_samples
            
            if self.syn_good_samples > 0:
                batch_indices.extend(syn_good_indices[syn_good_idx:syn_good_idx + self.syn_good_samples])
                syn_good_idx += self.syn_good_samples
            
            if self.syn_bad_samples > 0:
                batch_indices.extend(syn_bad_indices[syn_bad_idx:syn_bad_idx + self.syn_bad_samples])
                syn_bad_idx += self.syn_bad_samples
            
            # Shuffle the batch indices to avoid having all samples from one category together
            np.random.shuffle(batch_indices)
            yield batch_indices
    
    def __len__(self) -> int:
        return self.num_batches


def create_combined_balanced_dataloader(dataset, batch_size, real_ratio=0.5, good_ratio=0.5, 
                                        threshold=0.3, cache_dir='./', num_workers=4, drop_last=True):
    """
    Create a DataLoader with balanced sampling across both data source (real/synthetic) and grasp quality.
    
    Args:
        dataset: GraspEvalDataset instance
        batch_size: Size of each batch
        real_ratio: Ratio of real samples in each batch (0.0 to 1.0)
        good_ratio: Ratio of good grasps in each batch (0.0 to 1.0)
        threshold: Score threshold to classify grasps as good (>=threshold) or bad (<threshold)
        cache_dir: Directory to cache indices
        num_workers: Number of worker processes for data loading
        drop_last: Whether to drop the last incomplete batch
    
    Returns:
        DataLoader with combined balanced sampling
    """
    sampler = CombinedBatchSampler(
        dataset, 
        batch_size, 
        real_ratio, 
        good_ratio, 
        threshold,
        cache_dir,
        drop_last
    )
    
    return torch.utils.data.DataLoader(
        dataset, 
        batch_sampler=sampler,
        num_workers=num_workers,
        # Use spawn context if needed for multiprocessing
        # multiprocessing_context=multiprocessing.get_context("spawn")
    )

if __name__ == '__main__':
    
    acronym_root = '/home/seung/Workspaces/Datasets/ACRONYM/grasp_qnet'
    g1b_root = '/home/seung/Workspaces/Datasets/GraspNet-1Billion/grasp_qnet'
    split = 'train'
    dataset = GraspEvalDataset(g1b_root, acronym_root, split, use_normal=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    for i, data in enumerate(dataloader):
        obj_cloud, gripper_cloud, score = data
        obj_cloud = np.array(obj_cloud.numpy(), dtype=np.float32)[0]
        if obj_cloud.shape[1] == 6:
            normal = obj_cloud[:, 3:6]
            obj_cloud = obj_cloud[:, :3]
        gripper_cloud = np.array(gripper_cloud.numpy(), dtype=np.float32)
        score = score.numpy()
        # visualize 
        obj_cloud_inner_pcd = o3d.geometry.PointCloud()
        obj_cloud_inner_pcd.points = o3d.utility.Vector3dVector(obj_cloud)
        obj_cloud_inner_pcd.normals = o3d.utility.Vector3dVector(normal)
        
        obj_cloud_inner_pcd.paint_uniform_color([0, 1, 0]) # green
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        gripper_pcd = o3d.geometry.PointCloud()
        gripper_pcd.points = o3d.utility.Vector3dVector(gripper_cloud[0])
        gripper_pcd.paint_uniform_color([1, 0, 0]) # red
        o3d.visualization.draw_geometries([obj_cloud_inner_pcd, gripper_pcd, coord_frame])