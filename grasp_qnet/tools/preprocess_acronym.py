
import os
import sys
import numpy as np
import scipy.io as scio
from PIL import Image
import open3d as o3d
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


from graspnetAPI import GraspGroup, Grasp

sys.path.append('/home/seung/Workspaces/grasp/TestAdaGrasp/GraspQNet')
from data_utils import CameraInfo, create_point_cloud_from_depth_image, transform_point_cloud, remove_invisible_grasp_points

from graspnetAPI.utils.utils import get_obj_pose_list, generate_views, get_model_grasps, transform_points
from graspnetAPI.utils.rotation import batch_viewpoint_params_to_matrix
from graspnetAPI import GraspNet, GraspGroup

from torch.utils.data import DataLoader,ConcatDataset, WeightedRandomSampler
import h5py
import glob

import os
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
import h5py


def upsample_point_cloud(cloud, num_points):
    # Original points
    original_points = cloud
    num_original_points = original_points.shape[0]

    # If the original number of points is already more than or equal to desired points
    if num_original_points >= num_points:
        print("Point cloud already has more points than required. Consider downsampling instead.")
        return cloud
    
    # Generate new points by random sampling
    indices = np.random.choice(num_original_points, num_points - num_original_points, replace=True)
    new_points = original_points[indices]

    # Combine original points and new points
    return np.vstack((original_points, new_points))

class ModelFreeCollisionDetector():
    """ Collision detection in scenes without object labels. Current finger width and length are fixed.

        Input:
                scene_points: [numpy.ndarray, (N,3), numpy.float32]
                    the scene points to detect
                voxel_size: [float]
                    used for downsample

        Example usage:
            mfcdetector = ModelFreeCollisionDetector(scene_points, voxel_size=0.005)
            collision_mask = mfcdetector.detect(grasp_group, approach_dist=0.03)
            collision_mask, iou_list = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05, return_ious=True)
            collision_mask, empty_mask = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05,
                                            return_empty_grasp=True, empty_thresh=0.01)
            collision_mask, empty_mask, iou_list = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05,
                                            return_empty_grasp=True, empty_thresh=0.01, return_ious=True)
    """
    def __init__(self, scene_points, voxel_size=0.005):
        self.finger_width = 0.01
        self.finger_length = 0.06
        self.voxel_size = voxel_size
        scene_cloud = o3d.geometry.PointCloud()
        scene_cloud.points = o3d.utility.Vector3dVector(scene_points)
        scene_cloud = scene_cloud.voxel_down_sample(voxel_size)
        self.scene_points = np.array(scene_cloud.points)

    def detect(self, grasp_group, approach_dist=0.03, collision_thresh=0.05, return_empty_grasp=False, empty_thresh=0.01, return_ious=False):
        """ Detect collision of grasps.

            Input:
                grasp_group: [GraspGroup, M grasps]
                    the grasps to check
                approach_dist: [float]
                    the distance for a gripper to move along approaching direction before grasping
                    this shifting space requires no point either
                collision_thresh: [float]
                    if global collision iou is greater than this threshold,
                    a collision is detected
                return_empty_grasp: [bool]
                    if True, return a mask to imply whether there are objects in a grasp
                empty_thresh: [float]
                    if inner space iou is smaller than this threshold,
                    a collision is detected
                    only set when [return_empty_grasp] is True
                return_ious: [bool]
                    if True, return global collision iou and part collision ious
                    
            Output:
                collision_mask: [numpy.ndarray, (M,), numpy.bool]
                    True implies collision
                [optional] empty_mask: [numpy.ndarray, (M,), numpy.bool]
                    True implies empty grasp
                    only returned when [return_empty_grasp] is True
                [optional] iou_list: list of [numpy.ndarray, (M,), numpy.float32]
                    global and part collision ious, containing
                    [global_iou, left_iou, right_iou, bottom_iou, shifting_iou]
                    only returned when [return_ious] is True
        """
        approach_dist = max(approach_dist, self.finger_width)
        T = grasp_group.translations
        R = grasp_group.rotation_matrices
        heights = grasp_group.heights[:,np.newaxis]
        depths = grasp_group.depths[:,np.newaxis]
        widths = grasp_group.widths[:,np.newaxis]
        targets = self.scene_points[np.newaxis,:,:] - T[:,np.newaxis,:]
        targets = np.matmul(targets, R)

        ## collision detection
        # height mask
        mask1 = ((targets[:,:,2] > -heights/2) & (targets[:,:,2] < heights/2))
        # left finger mask
        mask2 = ((targets[:,:,0] > depths - self.finger_length) & (targets[:,:,0] < depths))
        mask3 = (targets[:,:,1] > -(widths/2 + self.finger_width))
        mask4 = (targets[:,:,1] < -widths/2)
        # right finger mask
        mask5 = (targets[:,:,1] < (widths/2 + self.finger_width))
        mask6 = (targets[:,:,1] > widths/2)
        # bottom mask
        mask7 = ((targets[:,:,0] <= depths - self.finger_length)\
                & (targets[:,:,0] > depths - self.finger_length - self.finger_width))
        # shifting mask
        mask8 = ((targets[:,:,0] <= depths - self.finger_length - self.finger_width)\
                & (targets[:,:,0] > depths - self.finger_length - self.finger_width - approach_dist))

        # get collision mask of each point
        left_mask = (mask1 & mask2 & mask3 & mask4)
        right_mask = (mask1 & mask2 & mask5 & mask6)
        bottom_mask = (mask1 & mask3 & mask5 & mask7)
        shifting_mask = (mask1 & mask3 & mask5 & mask8)
        global_mask = (left_mask | right_mask | bottom_mask | shifting_mask)

        # calculate equivalant volume of each part
        left_right_volume = (heights * self.finger_length * self.finger_width / (self.voxel_size**3)).reshape(-1)
        bottom_volume = (heights * (widths+2*self.finger_width) * self.finger_width / (self.voxel_size**3)).reshape(-1)
        shifting_volume = (heights * (widths+2*self.finger_width) * approach_dist / (self.voxel_size**3)).reshape(-1)
        volume = left_right_volume*2 + bottom_volume + shifting_volume

        # get collision iou of each part
        global_iou = global_mask.sum(axis=1) / (volume+1e-6)

        # get collison mask
        collision_mask = (global_iou > collision_thresh)

        if not (return_empty_grasp or return_ious):
            return collision_mask

        ret_value = [collision_mask,]
        if return_empty_grasp:
            inner_mask = (mask1 & mask2 & (~mask4) & (~mask6))
            inner_volume = (heights * self.finger_length * widths / (self.voxel_size**3)).reshape(-1)
            empty_mask = (inner_mask.sum(axis=-1)/inner_volume < empty_thresh)
            ret_value.append(empty_mask)
        if return_ious:
            left_iou = left_mask.sum(axis=1) / (left_right_volume+1e-6)
            right_iou = right_mask.sum(axis=1) / (left_right_volume+1e-6)
            bottom_iou = bottom_mask.sum(axis=1) / (bottom_volume+1e-6)
            shifting_iou = shifting_mask.sum(axis=1) / (shifting_volume+1e-6)
            ret_value.append([global_iou, left_iou, right_iou, bottom_iou, shifting_iou])
        return ret_value



def preprocess_data(grasp_path, acronym_path, out_root):
    with h5py.File(grasp_path, "r") as f:
        f = h5py.File(grasp_path, "r")
        mesh_fname = f["object/file"][()].decode('utf-8')
        
        
    mesh_path = os.path.join(acronym_path, mesh_fname)
    
    mesh_rescaled_path = mesh_path.replace('.obj', '_rescaled.obj')
    score_path = mesh_path.replace('.obj', '_scores.npy')
    gg_array_path = mesh_path.replace('.obj', '_gg.npy')
    if not os.path.exists(score_path) or not os.path.exists(gg_array_path):
        # print('Missing score or gg_array: ', mesh_fname)
        return

    mesh_fname = mesh_fname.replace('.obj', '')
    mesh_fname = mesh_fname.replace('meshes/', '')
    num_grasps = 100
    n_exists = glob.glob(out_root + f'/{mesh_fname}_*.h5')
    
    if len(n_exists) > num_grasps:
        print('Too many processed: ', mesh_fname)
        print('Number of processed grasps: ', len(n_exists))
        # remove the extra files
        for file in n_exists[num_grasps:]:
            os.remove(file)
        return
    
    if len(n_exists) == num_grasps:
        # print('Already processed: ', mesh_fname)
        return
    print('Number of processed grasps: ', len(n_exists))
    print(mesh_fname)
    num_grasps = num_grasps - len(n_exists)
    
    while True:
        n_exists = glob.glob(out_root + f'/{mesh_fname}_*.h5')
        if len(n_exists) >= 100:
            print('Already processed: ', mesh_fname)
            break
        else:
            print('trying to process: ', mesh_fname)
        
        # load gg_array
        gg = GraspGroup()
        gg.grasp_group_array = np.load(gg_array_path)
        # gg_o3d = gg.to_open3d_geometry_list()
        
        # load mesh
        mesh = o3d.io.read_triangle_mesh(mesh_rescaled_path)
        obj_pcd = mesh.sample_points_uniformly(2048000)
        obj_pcd = np.asarray(obj_pcd.points)
        
        # load score
        scores = np.load(score_path)
        
        mfcdetector = ModelFreeCollisionDetector(obj_pcd, voxel_size=0.01)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=0.01)
        
        # random downsample 100 grasps
        target_indices = list(range(gg.grasp_group_array.shape[0]))
        # target_indices = np.random.choice(gg.grasp_group_array.shape[0], num_grasps, replace=False)
        
        for target_idx in target_indices:
            n_exists = glob.glob(out_root + f'/{mesh_fname}_*.h5')
            if len(n_exists) >= 100:
                print('Already processed: ', mesh_fname)
                break
            save_path = os.path.join(out_root, f'{mesh_fname}_{target_idx:08d}.h5')
            if os.path.exists(save_path):
                print('Already processed: ', save_path)
                continue

            grasps = gg.grasp_group_array[target_idx].copy()
            grasps = np.reshape(grasps, [1, -1])
            
            score = scores[target_idx]
            score = max(1.1 - score, 0)
            if collision_mask[target_idx]:
                print('Collision detected: ', target_idx)
                score = 0.0
            ## parse grasp parameters
            grasp_points = grasps[:, 13:16]
            grasp_poses = grasps[:, 4:13].reshape([-1,3,3])
            grasp_depths = grasps[:, 3]
            grasp_widths = grasps[:, 1]

            # transform scene to gripper frame
            target = (obj_pcd[np.newaxis,:,:] - grasp_points[:,np.newaxis,:])
            target = np.matmul(target, grasp_poses)

            ## crop the object in gripper closing area
            height = 0.04
            depth_base = 0.04
            depth_outer = 0.04
            mask1 = ((target[:,:,2]>-height/2) & (target[:,:,2]<height/2))
            mask2 = ((target[:,:,0]>-depth_base - depth_outer) & (target[:,:,0]<grasp_depths[:,np.newaxis] + depth_outer))
            mask4 = (target[:,:,1]<-grasp_widths[:,np.newaxis]/2)
            mask6 = (target[:,:,1]>grasp_widths[:,np.newaxis]/2)
            inner_mask = (mask1 & mask2 &(~mask4) & (~mask6)) # [n_batch, n_points]
            obj_cloud_inner = obj_pcd[inner_mask[0]]

            num_cloud_points = 1024
            num_gripper_points = 128

            # random sample n_points
            if obj_cloud_inner.shape[0] >= num_cloud_points:
                obj_cloud_inner = obj_cloud_inner[np.random.choice(obj_cloud_inner.shape[0], num_cloud_points, replace=False)]
            elif obj_cloud_inner.shape[0] < 128:
                print('Not enough points: ', obj_cloud_inner.shape[0])
                continue
            else:
                obj_cloud_inner = upsample_point_cloud(obj_cloud_inner, num_cloud_points)


            # sample gripper points
            g = GraspGroup(grasps)[0]
            gripper_pcd = g.to_open3d_geometry().sample_points_poisson_disk(num_gripper_points)
            gripper_cloud = np.asarray(gripper_pcd.points)

            # align cloud to gripper frame
            se3 = np.eye(4)
            se3[:3,:3] = g.rotation_matrix.reshape(3,3)
            se3[:3,3] = g.translation
            se3 = np.linalg.inv(se3)
            gripper_pcd.transform(se3)
            obj_cloud_inner = np.matmul(obj_cloud_inner, se3[:3,:3].T) + se3[:3,3]
            
            # obj_cloud_inner_pcd = o3d.geometry.PointCloud()
            # obj_cloud_inner_pcd.points = o3d.utility.Vector3dVector(obj_cloud_inner)
            # obj_cloud_inner_pcd.paint_uniform_color([0, 1, 0]) # green
            # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            # o3d.visualization.draw_geometries([obj_cloud_inner_pcd, gripper_pcd, coord_frame])
            # exit()
            save_path = os.path.join(out_root, f'{mesh_fname}_{target_idx:08d}.h5')
            dir_name = os.path.dirname(save_path)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            with h5py.File(save_path, 'w') as f:
                f.create_dataset('obj_cloud', data=obj_cloud_inner)
                f.create_dataset('gripper_cloud', data=gripper_cloud)
                f.create_dataset('score', data=score)
            print('Saved: ', save_path)


def safe_process_func(grasp_path, acronym_path, out_root):
    try:
        return preprocess_data(grasp_path, acronym_path, out_root)
    except Exception as e:
        print(f"Error processing {grasp_path}: {e}")
        return None

if __name__ == "__main__":
    acronym_path = '/home/seung/Workspaces/Datasets/ACRONYM'
    
    grasp_dir_path = os.path.join(acronym_path, 'grasps')
    grasp_paths = glob.glob(grasp_dir_path + '/*.h5')
    grasp_paths = sorted(grasp_paths)
    
    # filter only 742e14aa241feab67efcee988123ee4c
    grasp_paths = [path for path in grasp_paths if '742e14aa241feab67efcee988123ee4c' in path]
    
    out_root = acronym_path + '/grasp_qnet'
    if not os.path.exists(out_root):
        os.makedirs(out_root)

    # Determine the number of processes
    num_processes = 1  # Leave one CPU free
    
    # Create a pool of workers - directly use the function without partial
    with mp.Pool(processes=num_processes) as pool:
        # Use starmap to pass multiple arguments to the function
        results = list(
            pool.starmap(
                safe_process_func,
                tqdm([(grasp_path, acronym_path, out_root) for grasp_path in grasp_paths], 
                     desc="Processing grasps",
                     total=len(grasp_paths))
            )
        )