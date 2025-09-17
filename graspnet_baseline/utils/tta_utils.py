import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import random
import time

from graspnetAPI import Grasp, GraspGroup
from tqdm import tqdm
from open3d.geometry import PointCloud 
from torch.utils.data import DataLoader
from dataset.graspnet_dataset import GraspNetDataset, collate_fn
from dataset.graspclutter6d_dataset import GraspClutter6DDataset



def jitter_point_cloud(cloud, sigmas=[0.001, 0.002]):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 tensor, original batch of point clouds
        Return:
          BxNx3 tensor, jittered batch of point clouds
    """
    sigma = np.random.uniform(sigmas[0], sigmas[1])
    cloud = cloud + sigma * torch.randn(cloud.shape).to(cloud)
    return cloud


def get_aug_matrix(type):
    if type == 'none':
        return np.eye(3)
    elif type == 'hflip':
        return np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    elif type == 'vflip':
        return np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    else:
        raise ValueError('Augmentation type not supported: %s' % type)

def augment_cloud(cloud, type='jitter'):
    """
    Augment point cloud with different methods.
        cloud = torch.tensor (B, N, 3)
    """
    if type == 'none':
        return cloud, None
    elif type == 'jitter':
        return jitter_point_cloud(cloud), None
    elif type == 'hflip': # flip along YZ plane
        flip_mat = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], device=cloud.device).float()
        # Handle batched point clouds
        return torch.matmul(cloud, flip_mat.T), flip_mat
    elif type == 'vflip': # flip along XZ plane
        flip_mat = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]], device=cloud.device).float()
        # Handle batched point clouds
        return torch.matmul(cloud, flip_mat.T), flip_mat
    elif type == 'rotate':
        B = cloud.shape[0]
        # Create batch of rotation matrices
        rot_matrices = []
        for i in range(B):
            # -30 to 30 degree rotation
            rot_angle = (np.random.random()*np.pi/3) - np.pi/6 # -30 ~ +30 degree
            c, s = np.cos(rot_angle), np.sin(rot_angle)
            rot_mat = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], device=cloud.device).float()
            rot_matrices.append(rot_mat)
        rot_tensors = torch.stack(rot_matrices)
        # Apply rotation to each point cloud in the batch
        rotated_clouds = []
        for i in range(B):
            rotated_clouds.append(torch.matmul(cloud[i], rot_tensors[i]))
        return torch.stack(rotated_clouds), rot_matrices
    else:
        raise ValueError('Augmentation type not supported: %s' % type)

def transform_point_cloud(cloud, mat):
    """
    Apply transformation matrix to point cloud.
        cloud = torch.tensor (N, 3)
        mat = torch.tensor (3, 3)
    """
    if mat is None:
        return cloud
    elif isinstance(cloud, torch.Tensor):
        return torch.matmul(mat, cloud.T).T
    elif isinstance(cloud, np.ndarray):
        # !TODO: check if this is correct
        mat = mat.cpu().numpy()
        # return np.dot(mat, cloud.T).T
        return np.matmul(cloud, mat)

def sample_point_cloud(point_cloud, target_points=1024):
    """
    Sample a point cloud to have exactly target_points.
    If the point cloud has more points than target_points, randomly downsample.
    If the point cloud has fewer points than target_points, randomly duplicate points.
    
    Args:
        point_cloud: Torch tensor of shape (N, 3) where:
            N is number of points
            3 is the dimension of each point (x, y, z)
        target_points: Target number of points (default: 1024)
    
    Returns:
        Torch tensor of shape (B, target_points, 3)
    """
    num_points, _ = point_cloud.shape
    
    if num_points >= target_points:
        # Downsample: randomly select target_points without replacement
        indices = torch.randperm(num_points, device=point_cloud.device)[:target_points]
        return point_cloud[indices]
    else:
        return torch.cat([point_cloud, point_cloud[
            torch.randint(0, num_points, (target_points - num_points,))]])




@torch.no_grad()
def ema_update_model(model_to_update, model_to_merge, momentum, device, update_all=True):
    if momentum < 1.0:
        for param_to_update, param_to_merge in zip(model_to_update.parameters(), model_to_merge.parameters()):
            if param_to_update.requires_grad or update_all:
                param_to_update.data = momentum * param_to_update.data + (1 - momentum) * param_to_merge.data.to(device)
    return model_to_update

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    # print(x.softmax(0))
    return -(x.softmax(0) * x.log_softmax(0)).sum(0)

def mse_loss(x, x_ema):
    return F.mse_loss(x, x_ema)



from collision_detector import ModelFreeCollisionDetector


class ModelFreeCollisionDetectorGPU():


    def __init__(self, scene_points, voxel_size=0.005, device='cuda'):
        self.finger_width = 0.01
        self.finger_length = 0.06
        self.voxel_size = voxel_size
        self.device = torch.device(device) # Use torch.device object

        # scene_points: torch.Tensor (N, 3)
        scene_cloud = o3d.geometry.PointCloud()
        if isinstance(scene_points, torch.Tensor):
            scene_points_np = scene_points.cpu().numpy()
        else:
            scene_points_np = scene_points
        scene_cloud.points = o3d.utility.Vector3dVector(scene_points_np)
        scene_cloud = scene_cloud.voxel_down_sample(voxel_size)
        self.scene_points = torch.tensor(np.asarray(scene_cloud.points), dtype=torch.float32, device=self.device)
        self.num_points = self.scene_points.shape[0]


    def detect(self, gg_array, approach_dist=0.03, collision_thresh=0.05,
               return_empty_grasp=False, empty_thresh=0.01, return_ious=False):
        """ Detect collision of grasps using optimized GPU acceleration. """
        approach_dist = max(approach_dist, self.finger_width)

        #[:, score, width, height, depth, rotation_matrix(9), translation(3), object_id]
        T_tensor = gg_array[:, 13:16]
        R_tensor = gg_array[:, 4:13].reshape(-1, 3, 3)
        heights_tensor = gg_array[:, 2].unsqueeze(1)
        depths_tensor = gg_array[:, 3].unsqueeze(1)
        widths_tensor = gg_array[:, 1].unsqueeze(1)
        # Convert to tensors
        num_grasps = T_tensor.shape[0]

        if num_grasps == 0:
             # Handle empty grasp group case
            ret_value = [np.array([], dtype=bool)]
            if return_empty_grasp:
                ret_value.append(np.array([], dtype=bool))
            if return_ious:
                ret_value.append([np.array([], dtype=np.float32)] * 5)
            return ret_value if len(ret_value) > 1 else ret_value[0]

        targets = self.scene_points.unsqueeze(0) - T_tensor.unsqueeze(1)
        transformed_targets = torch.einsum('bpi,bij->bpj', targets, R_tensor) # einsum is fine
        # Collision detection masks (compute in a single large batch)
        # Height mask
        mask1 = (transformed_targets[:,:,2] > -heights_tensor/2) & (transformed_targets[:,:,2] < heights_tensor/2)

        # Left finger region
        left_region = (transformed_targets[:,:,0] > depths_tensor - self.finger_length) & \
                      (transformed_targets[:,:,0] < depths_tensor) & \
                      (transformed_targets[:,:,1] > -(widths_tensor/2 + self.finger_width)) & \
                      (transformed_targets[:,:,1] < -widths_tensor/2)
        # Right finger region
        right_region = (transformed_targets[:,:,0] > depths_tensor - self.finger_length) & \
                       (transformed_targets[:,:,0] < depths_tensor) & \
                       (transformed_targets[:,:,1] < (widths_tensor/2 + self.finger_width)) & \
                       (transformed_targets[:,:,1] > widths_tensor/2)
        # Bottom region condition
        bottom_region_cond = (transformed_targets[:,:,0] <= depths_tensor - self.finger_length) & \
                         (transformed_targets[:,:,0] > depths_tensor - self.finger_length - self.finger_width)
        # Shifting region condition
        shifting_region_cond = (transformed_targets[:,:,0] <= depths_tensor - self.finger_length - self.finger_width) & \
                           (transformed_targets[:,:,0] > depths_tensor - self.finger_length - self.finger_width - approach_dist)
        # Shared width region for bottom and shifting
        width_region_both = (transformed_targets[:,:,1] > -(widths_tensor/2 + self.finger_width)) & \
                            (transformed_targets[:,:,1] < (widths_tensor/2 + self.finger_width))
        # Final Masks
        left_mask = mask1 & left_region
        right_mask = mask1 & right_region
        bottom_mask = mask1 & bottom_region_cond & width_region_both
        shifting_mask = mask1 & shifting_region_cond & width_region_both
        # Combine all masks to get global collision mask
        global_mask = left_mask | right_mask | bottom_mask | shifting_mask
        # Calculate equivalent volume of each part (on GPU)
        # Ensure shapes are compatible for broadcasting, squeeze unnecessary dims
        left_right_volume = (heights_tensor * self.finger_length * self.finger_width / (self.voxel_size**3)).squeeze(-1) # Shape [num_grasps]
        bottom_volume = (heights_tensor * (widths_tensor+2*self.finger_width) * self.finger_width / (self.voxel_size**3)).squeeze(-1) # Shape [num_grasps]
        shifting_volume = (heights_tensor * (widths_tensor+2*self.finger_width) * approach_dist / (self.voxel_size**3)).squeeze(-1) # Shape [num_grasps]
        # Ensure volumes are non-zero before division, add epsilon
        epsilon = 1e-6
        volume = left_right_volume * 2 + bottom_volume + shifting_volume + epsilon
        # Calculate IOUs (on GPU)
        # Sum over the points dimension (dim=1)
        global_iou = torch.sum(global_mask.float(), dim=1) / volume
        # Get collision mask (on GPU)
        collision_mask_tensor = (global_iou > collision_thresh)
        # --- Prepare results ---
        # Transfer final results back to CPU ONCE
        collision_mask = collision_mask_tensor.cpu().numpy()
        ret_value = [collision_mask,]
        if return_empty_grasp:
            # Inner region for empty grasp detection
            inner_region = (transformed_targets[:,:,0] > depths_tensor - self.finger_length) & \
                           (transformed_targets[:,:,0] < depths_tensor) & \
                           (transformed_targets[:,:,1] > -widths_tensor/2) & \
                           (transformed_targets[:,:,1] < widths_tensor/2)
            inner_mask = mask1 & inner_region
            # Ensure inner_volume has shape [num_grasps]
            inner_volume = (heights_tensor * self.finger_length * widths_tensor / (self.voxel_size**3)).squeeze(-1) + epsilon
            empty_mask_tensor = (torch.sum(inner_mask.float(), dim=1) / inner_volume < empty_thresh)
            ret_value.append(empty_mask_tensor.cpu().numpy())

        if return_ious:
            left_iou = torch.sum(left_mask.float(), dim=1) / (left_right_volume + epsilon)
            right_iou = torch.sum(right_mask.float(), dim=1) / (left_right_volume + epsilon)
            bottom_iou = torch.sum(bottom_mask.float(), dim=1) / (bottom_volume + epsilon)
            shifting_iou = torch.sum(shifting_mask.float(), dim=1) / (shifting_volume + epsilon)
            iou_list_cpu = [
                global_iou.cpu().numpy(),
                left_iou.cpu().numpy(),
                right_iou.cpu().numpy(),
                bottom_iou.cpu().numpy(),
                shifting_iou.cpu().numpy()
            ]
            ret_value.append(iou_list_cpu)
        # Return only the mask if no other options are selected, otherwise the list
        return ret_value[0] if len(ret_value) == 1 else ret_value

def analytical_sampling(scene_cloud, cfg_path, num_grasp_samples=1000, foreground_cloud=None):
    """Analytical sampling of grasps from a point cloud using GPG
    """
    scene_cloud = scene_cloud.cpu().numpy()
    scene_cloud_o3d = o3d.geometry.PointCloud()
    scene_cloud_o3d.points = o3d.utility.Vector3dVector(scene_cloud)
    scene_cloud_o3d.voxel_down_sample(voxel_size=0.01)
    scene_cloud = np.asarray(scene_cloud_o3d.points)

    if foreground_cloud is not None and len(foreground_cloud) > 0:
        foreground_cloud = foreground_cloud.cpu().numpy()
        # remove only nearest scene_cloud points with distance less than 0.01
        scene_cloud = torch.from_numpy(scene_cloud).float().cuda()
        foreground_cloud = torch.from_numpy(foreground_cloud).float().cuda()
        dist = torch.norm(scene_cloud[:, None, :] - foreground_cloud[None, :, :], dim=-1)
        scene_cloud_foreground = scene_cloud[torch.min(dist, dim=-1)[0] < 0.01]
        if len(scene_cloud_foreground) > 10000:
            scene_cloud_foreground = scene_cloud_foreground[torch.randperm(scene_cloud_foreground.shape[0])[:10000]]
        scene_cloud_foreground = scene_cloud_foreground.cpu().numpy()
        start_time = time.time()
        grasps = pygpg.generate_grasps(scene_cloud_foreground, num_grasp_samples, False, cfg_path)
        print('--> grasp time: ', time.time() - start_time)
    else:
        if len(scene_cloud) > 10000:
            scene_cloud = scene_cloud[np.random.choice(scene_cloud.shape[0], 10000, replace=False)]
        grasps = pygpg.generate_grasps(scene_cloud, num_grasp_samples, False, cfg_path)
    gg_array = []
    for grasp in grasps:
        bottom = grasp.get_grasp_bottom()
        top = grasp.get_grasp_top()
        approach = grasp.get_grasp_approach()
        binormal = grasp.get_grasp_binormal()
        axis = grasp.get_grasp_axis()
        width = grasp.get_grasp_width()
        pose = np.eye(4)
        pose[:3, 0] = approach
        pose[:3, 1] = binormal
        pose[:3, 2] = axis
        contact = (top + bottom) / 2
        pose[:3, 3] = contact
        width = grasp.get_grasp_width()*2
        g_array = np.array([
            0.1, width, 0.02, 0.02, *pose[:3, :3].reshape(-1), *pose[:3, 3], -1
        ])
        gg_array.append(g_array)
    gg_array = np.array(gg_array)
    gg = GraspGroup(gg_array)
    # gg = gg.nms(0.03, 30.0/180*np.pi)
    if isinstance(scene_cloud, torch.Tensor):
        scene_cloud = scene_cloud.cpu().numpy()
    mfcdetector = ModelFreeCollisionDetector(scene_cloud, voxel_size=0.01)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=0.01)
    gg_array = gg.grasp_group_array[~collision_mask]
    
    new_gg_array = []
    for grasp in gg_array:
        rotation = grasp[4:13].reshape(3, 3)
        # check the angle with z-axis (camera frame) vs [0, 0, 1]
        angle = np.dot(rotation[:3, 0] / np.linalg.norm(rotation[:3, 0]), 
                       np.array([0, 0, 1]))
        angle = np.clip(angle, -1, 1)
        angle = np.arccos(angle)
        if np.rad2deg(angle) < 60:
            new_gg_array.append(grasp)
    gg_array = np.array(new_gg_array)
    gg = GraspGroup(gg_array)
    gg_array = torch.from_numpy(gg_array).float().cuda()

    # gg_o3d = gg.to_open3d_geometry_list()
    # points_o3d = o3d.geometry.PointCloud()
    # points_o3d.points = o3d.utility.Vector3dVector(scene_cloud)
    # o3d.visualization.draw_geometries([points_o3d] + gg_o3d)
    # # print(f"=======> Time taken: {time.time() - start_time} seconds")
    return gg_array


def get_index_A_to_B(gg_A, gg_B):
    '''Get the index of gg_A in gg_B'''
    # gg_A: [N, 17], gg_B: [M, 17]
    # return index of A in B in dim 0
    gg_A = gg_A[:, 1:16].cpu().numpy() # do not use score and object_id for matching
    gg_B = gg_B[:, 1:16].cpu().numpy()
    
    # Create a dictionary for quick lookups of gg_B rows
    gg_B_dict = {tuple(row): idx for idx, row in enumerate(gg_B)}
    
    index = []
    for i in range(gg_A.shape[0]):
        row = tuple(gg_A[i])
        if row in gg_B_dict:
            index.append(gg_B_dict[row])
    
    return index


def crop_inner_cloud(gg_array, scene_cloud, min_points=512, nms_on=False, num_cloud_points=1024, num_gripper_points=128):
    """_summary_

    Args:
        GraspGroup (gg): _description_

    Returns:
        _type_: _description_
    """
    # ## 1. first apply grasp NMS 
    if nms_on:
        gg = GraspGroup()
        gg.grasp_group_array = gg_array.cpu().numpy()
        gg = gg.nms(0.03, 30.0/180*np.pi)
        gg_array = torch.from_numpy(np.array(gg.grasp_group_array, copy=True)).float().cuda()

    # 2. detect collision (GPU for speed)
    mfcdetector = ModelFreeCollisionDetectorGPU(scene_cloud, voxel_size=0.01)
    collision_mask = mfcdetector.detect(gg_array, approach_dist=0.05, collision_thresh=0.01)
    gg_array = gg_array[~collision_mask]

    # 3. crop point clouds inside of gripper
    height = 0.04
    depth_base = 0.04
    depth_outer = 0.04
    # Parse grasp parameters
    grasp_points = gg_array[:, 13:16]  # (N, 3)
    grasp_poses = gg_array[:, 4:13].reshape(-1, 3, 3)  # (N, 3, 3)
    grasp_depths = gg_array[:, 3]  # (N,)
    grasp_widths = gg_array[:, 1]  # (N,)
    # Compute the target points for all grasps
    scene_cloud_expanded = scene_cloud.unsqueeze(0).expand(grasp_points.shape[0], -1, -1)  # (N, M, 3)
    target = scene_cloud_expanded - grasp_points.unsqueeze(1)  # (N, M, 3)
    target = torch.bmm(target, grasp_poses.transpose(1, 2))  # (N, M, 3)
    # Crop the object in gripper closing area
    mask1 = (target[:, :, 2] > -height / 2) & (target[:, :, 2] < height / 2)
    mask2 = (target[:, :, 0] > -depth_base) & (target[:, :, 0] < grasp_depths.unsqueeze(1) + depth_outer)
    mask4 = target[:, :, 1] < -grasp_widths.unsqueeze(1) / 2
    mask6 = target[:, :, 1] > grasp_widths.unsqueeze(1) / 2
    inner_mask = mask1 & mask2 & (~mask4) & (~mask6)  # (N, M)
    valid_indices = torch.where(inner_mask.sum(dim=1) > min_points)[0] 
    inner_mask = inner_mask[valid_indices]
    grasp_points = grasp_points[valid_indices]
    grasp_poses = grasp_poses[valid_indices]
    grasp_depths = grasp_depths[valid_indices]
    grasp_widths = grasp_widths[valid_indices]
    gg_array = gg_array[valid_indices]
    gg_array_np = gg_array.cpu().numpy()

    # 4. Prepare input for GEvalNet
    gripper_clouds = []
    obj_clouds = []
    g_list = [Grasp(gg) for gg in gg_array_np]
    se3_batch = torch.eye(4).cuda().unsqueeze(0).repeat(grasp_points.shape[0], 1, 1)
    se3_batch[:, :3, :3] = grasp_poses  # Assuming grasp_poses can be assigned as a batch
    se3_batch[:, :3, 3] = grasp_points
    se3_batch = torch.inverse(se3_batch)  # Batch inverse
    for i in range(grasp_points.shape[0]):
        se3 = se3_batch[i]
        gripper_pcd = g_list[i].to_open3d_geometry().sample_points_uniformly(num_gripper_points)
        gripper_pcd.transform(se3.cpu().numpy())
        gripper_clouds.append(torch.from_numpy(np.asarray(gripper_pcd.points)).float().cuda())
        obj_cloud_inner = sample_point_cloud(scene_cloud[inner_mask[i]], num_cloud_points)
        obj_cloud_inner = torch.matmul(obj_cloud_inner, se3[:3, :3].T) + se3[:3, 3]
        obj_clouds.append(obj_cloud_inner)

    if len(gripper_clouds) == 0:
        return [], [], gg_array

    gripper_clouds = torch.stack(gripper_clouds).float().cuda()
    obj_clouds = torch.stack(obj_clouds).float().cuda()
   
    # visualzie obj_clouds and gripper_clouds
    # obj_clouds_o3d = o3d.geometry.PointCloud()
    # obj_clouds_o3d.points = o3d.utility.Vector3dVector(obj_clouds[0].cpu().numpy())
    # gripper_clouds_o3d = o3d.geometry.PointCloud()
    # gripper_clouds_o3d.points = o3d.utility.Vector3dVector(gripper_clouds[0].cpu().numpy())
    # o3d.visualization.draw_geometries([obj_clouds_o3d, gripper_clouds_o3d])
    return obj_clouds, gripper_clouds, gg_array

def load_optimizer(optimizer_name, model_parameters, lr, weight_decay=0.0):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        return torch.optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f'Optimizer {optimizer_name} not supported')