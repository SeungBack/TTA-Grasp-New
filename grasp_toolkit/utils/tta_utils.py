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

from .collision_detector import ModelFreeCollisionDetector, ModelFreeCollisionDetectorGPU
from .data_utils import sample_point_cloud



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


def crop_inner_cloud(gg_array, scene_cloud, min_points=512, nms_on=False, num_cloud_points=1024, num_gripper_points=128, max_grasp_num=512):
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
    # sort
    # gg_array = gg_array[torch.argsort(gg_array[:, 0], descending=True)][:max_grasp_num]
    # # use only top 512
    # gg_array = gg_array[:512]
    
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
    if scene_cloud.shape[0] > 400000: # to prevent GPU OOM
        scene_cloud = sample_point_cloud(scene_cloud, 400000)
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

def load_optimizer(optimizer_name, model, lr, backbone_lr_ratio=1.0, weight_decay=0.0):
    
    if optimizer_name == "adamw":

        if backbone_lr_ratio == -1.0:
            # train only the backbone
            backbone_params = [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad]
            return torch.optim.AdamW(backbone_params, lr=lr, weight_decay=weight_decay)

        elif backbone_lr_ratio != 1.0:
            # train backbone and head with different lr
            backbone_params = [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad]
            non_backbone_params = [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad]
            return torch.optim.AdamW([{'params': backbone_params, 'lr': lr * backbone_lr_ratio},
                                    {'params': non_backbone_params, 'lr': lr}], weight_decay=weight_decay)
        else:
            # params = []
            # for nm, m in model.named_modules():
            #     if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            #         for np, p in m.named_parameters():
            #             if np in ['weight', 'bias']:  # weight is scale, bias is shift
            #                 params.append(p)
            return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)

    else:
        raise ValueError(f'Optimizer {optimizer_name} not supported')