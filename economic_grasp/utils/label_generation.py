import os
import sys
import time
import pdb

import torch
import os
import sys
import torch
import math

from libs.knn.knn_modules import knn
from utils.loss_utils import (batch_viewpoint_params_to_matrix, transform_point_cloud,
                              generate_grasp_views, compute_pointwise_dists)
from utils.arguments import cfgs


# def process_grasp_pseudo_label(end_points):
#     """ Process labels according to scene points and object poses. """
#     seed_xyzs = end_points['xyz_graspable']  # [B (batch size), 1024 (scene points after sample), 3]
#     pred_top_view_inds = end_points['grasp_top_view_inds']  # [B (batch size), 1024 (scene points after sample)]
#     batch_size, num_samples, _ = seed_xyzs.size()

#     valid_points_count = 0
#     valid_views_count = 0

#     batch_grasp_points = []
#     batch_grasp_views_rot = []
#     batch_view_graspness = []
#     batch_grasp_rotations = []
#     batch_grasp_depth = []
#     batch_grasp_scores = []
#     batch_grasp_widths = []
#     batch_valid_mask = []
    
#     mat_aug_student = end_points['mat_aug_student'] if 'poses_student' in end_points else None
    
    
#     for i in range(batch_size):
#         seed_xyz = seed_xyzs[i]  # [1024 (scene points after sample), 3]
#         pred_top_view = pred_top_view_inds[i]  # [1024 (scene points after sample)]
    
#         gg_array_ema = end_points['batch_grasp_preds_ema'][i] #(N, 17)
#         grasp_top_view_inds_ema = end_points['batch_grasp_top_view_inds_ema'][i] #(N,)
#         grasp_angles_ema = end_points['batch_grasp_angles_ema'][i] #(N)
#         mat_aug = end_points['mat_aug'][i] #(N, 3, 3)
#         # grasp_q = end_points['batch_grasp_q'][i] #(N, 1)
#         view_scores_ema = end_points['batch_view_scores_ema'][i] # (N, V)

#         # get merged grasp points for label computation
#         # transform the view from object coordinate system to scene coordinate system
#         grasp_points_merged = []
#         grasp_views_rot_merged = []
#         grasp_rotations_merged = []
#         grasp_depth_merged = []
#         grasp_scores_merged = []
#         grasp_widths_merged = []
#         view_graspness_merged = []
#         top_view_index_merged = []

#         N = len(gg_array_ema)
#         V, A, D = cfgs.num_view, 12, 4
#         device = seed_xyz.device
        
#         grasp_scores_all = gg_array_ema[:, 0]  # (N,)
#         grasp_widths_all = gg_array_ema[:, 1]  # (N,)
#         grasp_depths_all = gg_array_ema[:, 3]  # (N,)
#         grasp_points_all = gg_array_ema[:, 13:16]  # (N, 3)
        
#         # Create pose matrices for all objects
#         poses = torch.eye(4, device=device).unsqueeze(0).repeat(N, 1, 1)  # (N, 4, 4)
#         for j, mat in enumerate(mat_aug):
#             if mat is not None:
#                 poses[j, :3, :3] = mat
                
#         poses_student = torch.eye(4, device=device).unsqueeze(0).repeat(N, 1, 1)  # (N, 4, 4)
#         poses_student[:, :3, :3] = torch.inverse(mat_aug_student) if mat_aug_student is not None else torch.eye(3, device=device).unsqueeze(0).repeat(N, 1, 1)
                
#         ###################################################
        
#         # Initialize grasp tensors for all predictions
#         grasp_scores = torch.zeros(N, V, A, D, dtype=torch.float32, device=device)
#         grasp_widths = torch.zeros(N, V, A, D, dtype=torch.float32, device=device)
#         view_graspness = view_scores_ema  # (N, V) - already in correct format

#         # Compute indices for all predictions
#         angle_inds = (grasp_angles_ema * A / torch.pi).long()  # (N,)
#         depth_inds = ((grasp_depths_all / 0.01) - 1).long()  # (N,)

#         # Fill grasp tensors using vectorized indexing
#         batch_inds = torch.arange(N, device=device)
#         grasp_scores[batch_inds, grasp_top_view_inds_ema, angle_inds, depth_inds] = grasp_scores_all
#         grasp_widths[batch_inds, grasp_top_view_inds_ema, angle_inds, depth_inds] = grasp_widths_all

#         # Vectorized depth and angle selection (same as original logic)
#         grasp_score_label_max_depth, grasp_score_label_max_depth_idx = grasp_scores.max(-1)
#         grasp_widths = grasp_widths.gather(-1, grasp_score_label_max_depth_idx.unsqueeze(-1)).squeeze(-1)
        
#         grasp_score_label_max_angle, grasp_score_label_max_angle_idx = grasp_score_label_max_depth.max(-1)
#         grasp_depths = grasp_score_label_max_depth_idx.gather(-1, grasp_score_label_max_angle_idx.unsqueeze(-1)).squeeze(-1)
#         grasp_rotations = grasp_score_label_max_angle_idx  # [N, V]
#         grasp_scores = grasp_score_label_max_angle  # [N, V]
#         grasp_widths = grasp_widths.gather(-1, grasp_score_label_max_angle_idx.unsqueeze(-1)).squeeze(-1)

#         # Select top views for all predictions
#         values, top_view_index = torch.topk(view_graspness, k=V)
#         grasp_rotations = torch.gather(grasp_rotations, 1, top_view_index)
#         grasp_depths = torch.gather(grasp_depths, 1, top_view_index)
#         grasp_scores = torch.gather(grasp_scores, 1, top_view_index)
#         grasp_widths = torch.gather(grasp_widths, 1, top_view_index)

#         # Generate template views (shared across all predictions)
#         grasp_views = generate_grasp_views(V).to(device)  # [V, 3]

#         # Transform grasp points for all predictions
#         grasp_points_trans_list = []
#         for n in range(N):
#             pts_trans = transform_point_cloud(grasp_points_all[n:n+1], poses[n], '3x4')
#             grasp_points_trans_list.append(pts_trans)
#         grasp_points_trans = torch.cat(grasp_points_trans_list, dim=0)

#         # Transform grasp views for all predictions  
#         grasp_views_expanded = grasp_views.unsqueeze(0).expand(N, -1, -1)  # (N, V, 3)
#         grasp_views_trans = torch.bmm(poses[:, :3, :3], grasp_views_expanded.transpose(1, 2)).transpose(1, 2)  # (N, V, 3)
        
#         grasp_views_trans = torch.bmm(poses_student[:, :3, :3], grasp_views_expanded.transpose(1, 2)).transpose(1, 2)  # (N, V, 3)

#         # Generate view rotation matrices for all predictions
#         angles = torch.zeros(N, V, dtype=grasp_views.dtype, device=device)
#         try:
#             # Try batch operation first
#             grasp_views_rot = batch_viewpoint_params_to_matrix(-grasp_views_expanded, angles)  # (N, V, 3, 3)
#         except:
#             # Fallback to sequential processing if batch not supported
#             grasp_views_rot = torch.stack([
#                 batch_viewpoint_params_to_matrix(-grasp_views_expanded[n], angles[n]) 
#                 for n in range(N)
#             ], dim=0)

#         # Transform rotation matrices
#         grasp_views_rot_trans = torch.matmul(poses[:, :3, :3].unsqueeze(1), grasp_views_rot)  # (N, V, 3, 3)
#         grasp_views_rot_trans = torch.matmul(poses_student[:, :3, :3].unsqueeze(1), grasp_views_rot)  # (N, V, 3, 3)

#         # ================== VIEW ASSIGNMENT (KNN MATCHING) ==================
        
#         # Prepare reference views (same for all predictions)
#         grasp_views_ref = grasp_views.transpose(0, 1).contiguous()  # (3, V)
        
#         # Process each prediction's view assignment (KNN requires individual processing)
#         view_graspness_trans_list = []
#         grasp_views_rot_trans_reordered_list = []
#         top_view_index_trans_list = []
        
#         for n in range(N):
#             # Transform views for this prediction
#             grasp_views_trans_n = grasp_views_trans[n].transpose(0, 1).contiguous().unsqueeze(0)  # (1, 3, V)
#             grasp_views_ref_expanded = grasp_views_ref.unsqueeze(0)  # (1, 3, V)
            
#             # Find nearest neighbor views
#             view_inds_raw = knn(grasp_views_trans_n, grasp_views_ref_expanded, k=1).squeeze().squeeze()
#             view_inds = view_inds_raw - 1  # [V]
            
#             # Transform view graspness
#             view_graspness_trans = torch.index_select(view_graspness[n:n+1], 1, view_inds)  # [1, V]
            
#             # Transform rotation matrices
#             grasp_views_rot_trans_n = torch.index_select(grasp_views_rot_trans[n], 0, view_inds)
#             grasp_views_rot_trans_n = grasp_views_rot_trans_n.unsqueeze(0)  # [1, V, 3, 3]
            
#             # Create top view index mapping
#             top_view_index_trans = (-1 * torch.ones((1, grasp_rotations.shape[1]), dtype=torch.long, device=device))
#             view_inds_3d = view_inds.unsqueeze(0).unsqueeze(-1)  # [1, V, 1] 
#             top_view_3d = top_view_index[n].unsqueeze(0).unsqueeze(0)  # [1, 1, V]
#             matches = (view_inds_3d == top_view_3d)  # [1, V, V] - 3D tensor
#             tpid, tvip, tids = torch.where(matches)  # Now returns 3 values as expected
#             if len(tids) > 0:
#                 top_view_index_trans[tpid, tvip] = tids
            
#             view_graspness_trans_list.append(view_graspness_trans)
#             grasp_views_rot_trans_reordered_list.append(grasp_views_rot_trans_n)
#             top_view_index_trans_list.append(top_view_index_trans)

#         # Combine results from all predictions
#         grasp_points_merged = grasp_points_trans  # [N, 3]
#         view_graspness_merged = torch.cat(view_graspness_trans_list, dim=0)  # [N, V]
#         top_view_index_merged = torch.cat(top_view_index_trans_list, dim=0)  # [N, V]
#         grasp_rotations_merged = grasp_rotations.to(torch.int32)  # [N, V]
#         grasp_depth_merged = grasp_depths.to(torch.int32)  # [N, V]
#         grasp_scores_merged = grasp_scores  # [N, V]
#         grasp_widths_merged = grasp_widths  # [N, V]
#         grasp_views_rot_merged = torch.cat(grasp_views_rot_trans_reordered_list, dim=0)  # [N, V, 3, 3]
#         # ================== ASSIGNMENT TO SCENE POINTS (KNN MATCHING) =================

#         # compute nearest neighbors
#         seed_xyz_ = seed_xyz.transpose(0, 1).contiguous().unsqueeze(0)
#         grasp_points_merged_ = grasp_points_merged.transpose(0, 1).contiguous().unsqueeze(0)
#         nn_inds = knn(grasp_points_merged_, seed_xyz_, k=1).squeeze() - 1

#         # assign anchor points to real points
#         grasp_points_merged = torch.index_select(grasp_points_merged, 0, nn_inds)
#         # [1024 (scene points after sample), 3]
#         grasp_views_rot_merged = torch.index_select(grasp_views_rot_merged, 0, nn_inds)
#         # [1024 (scene points after sample), 300, 3, 3]
#         view_graspness_merged = torch.index_select(view_graspness_merged, 0, nn_inds)
#         # [1024 (scene points after sample), 300]
#         top_view_index_merged = torch.index_select(top_view_index_merged, 0, nn_inds)
#         # [1024 (scene points after sample), num_of_view]
#         grasp_rotations_merged = torch.index_select(grasp_rotations_merged, 0, nn_inds)
#         # [1024 (scene points after sample), num_of_view]
#         grasp_depth_merged = torch.index_select(grasp_depth_merged, 0, nn_inds)
#         # [1024 (scene points after sample), num_of_view]
#         grasp_scores_merged = torch.index_select(grasp_scores_merged, 0, nn_inds)
#         # [1024 (scene points after sample), num_of_view]
#         grasp_widths_merged = torch.index_select(grasp_widths_merged, 0, nn_inds)
#         # [1024 (scene points after sample), num_of_view]

#         # select top view's rot, score and width
#         # we only assign labels when the pred view is in the pre-defined 60 top view, others are zero
#         pred_top_view_ = pred_top_view.view(num_samples, 1, 1, 1).expand(-1, -1, 3, 3)
#         # [1024 (points after sample), 1, 3, 3]
#         top_grasp_views_rot = torch.gather(grasp_views_rot_merged, 1, pred_top_view_).squeeze(1)
#         # [1024 (points after sample), 3, 3]
#         pid, vid = torch.where(pred_top_view.unsqueeze(-1) == top_view_index_merged)
#         # both pid and vid are [true numbers], where(condition) equals to nonzero(condition)
#         top_grasp_rotations = 12 * torch.ones(num_samples, dtype=torch.int32).to(seed_xyz.device)
#         # [1024 (points after sample)]
#         top_grasp_depth = 4 * torch.ones(num_samples, dtype=torch.int32).to(seed_xyz.device)
#         # [1024 (points after sample)]
#         top_grasp_scores = torch.zeros(num_samples, dtype=torch.float32).to(seed_xyz.device)
#         # [1024 (points after sample)]
#         top_grasp_widths = 0.1 * torch.ones(num_samples, dtype=torch.float32).to(seed_xyz.device)
#         # [1024 (points after sample)]
#         top_grasp_rotations[pid] = torch.gather(grasp_rotations_merged[pid], 1, vid.view(-1, 1)).squeeze(1)
#         top_grasp_depth[pid] = torch.gather(grasp_depth_merged[pid], 1, vid.view(-1, 1)).squeeze(1)
#         top_grasp_scores[pid] = torch.gather(grasp_scores_merged[pid], 1, vid.view(-1, 1)).squeeze(1)
#         top_grasp_widths[pid] = torch.gather(grasp_widths_merged[pid], 1, vid.view(-1, 1)).squeeze(1)

#         # only compute loss in the points with correct matching (so compute the mask first)
#         dist = compute_pointwise_dists(seed_xyz, grasp_points_merged)
#         valid_point_mask = dist < 0.005
#         valid_view_mask = torch.zeros(num_samples, dtype=torch.bool).to(seed_xyz.device)
#         valid_view_mask[pid] = True
#         valid_points_count = valid_points_count + torch.sum(valid_point_mask)
#         valid_views_count = valid_views_count + torch.sum(valid_view_mask)
#         valid_score_mask = top_grasp_scores > 0.0
#         # print(torch.sum(valid_point_mask), torch.sum(valid_view_mask), torch.sum(valid_score_mask))
#         valid_mask = valid_score_mask  & valid_view_mask  & valid_point_mask
#         # valid_mask = valid_view_mask  & valid_point_mask

#         # add to batch
#         batch_grasp_points.append(grasp_points_merged)
#         batch_grasp_views_rot.append(top_grasp_views_rot)
#         batch_view_graspness.append(view_graspness_merged)
#         batch_grasp_rotations.append(top_grasp_rotations)
#         batch_grasp_depth.append(top_grasp_depth)
#         batch_grasp_scores.append(top_grasp_scores)
#         batch_grasp_widths.append(top_grasp_widths)
#         batch_valid_mask.append(valid_mask)

#     batch_grasp_points = torch.stack(batch_grasp_points, 0)
#     # [B (batch size), 1024 (scene points after sample), 3]
#     batch_grasp_views_rot = torch.stack(batch_grasp_views_rot, 0)
#     # [B (batch size), 1024 (scene points after sample), 3, 3]
#     batch_view_graspness = torch.stack(batch_view_graspness, 0)
#     # [B (batch size), 1024 (scene points after sample), 300]
#     batch_grasp_rotations = torch.stack(batch_grasp_rotations, 0)
#     # [B (batch size), 1024 (scene points after sample)]
#     batch_grasp_depth = torch.stack(batch_grasp_depth, 0) # 0~4 
#     # [B (batch size), 1024 (scene points after sample)]
#     batch_grasp_scores = torch.stack(batch_grasp_scores, 0)
#     # [B (batch size), 1024 (scene points after sample)]
#     batch_grasp_widths = torch.stack(batch_grasp_widths, 0)
#     # [B (batch size), 1024 (scene points after sample)]
#     batch_valid_mask = torch.stack(batch_valid_mask, 0)
#     # [B (batch size), 1024 (scene points after sample)]
    
#     # print(batch_valid_mask.sum(), batch_valid_mask.shape, batch_valid_mask.dtype)
#     # # visualize the grasps
#     import numpy as np
#     import open3d as o3d
#     from graspnetAPI.utils.utils import plot_gripper_pro_max
#     grippers = []
#     print(len(batch_grasp_points[0][batch_valid_mask[0]]), torch.sum(batch_valid_mask[0]))
#     for i in range(len(batch_grasp_points[0][batch_valid_mask[0]])):

#         t = batch_grasp_points[0][batch_valid_mask[0]][i].detach().cpu().numpy()
#         R = batch_grasp_views_rot[0][batch_valid_mask[0]][i].detach().cpu().numpy()
#         width = batch_grasp_widths[0][batch_valid_mask[0]][i].detach().cpu().numpy()
#         depth = (batch_grasp_depth[0][batch_valid_mask[0]][i].detach().cpu().numpy() + 1) * 0.01
#         score = batch_grasp_scores[0][batch_valid_mask[0]][i].detach().cpu().numpy()
#         gripper = plot_gripper_pro_max(t, R, width, depth, score)
#         grippers.append(gripper)
#         # if i > 100:
#         #     break
#     scene_cloud = end_points['point_clouds_raw'][0].detach().cpu().numpy()
#     scene_cloud_o3d = o3d.geometry.PointCloud()
#     scene_cloud_o3d.points = o3d.utility.Vector3dVector(scene_cloud)
#     o3d.visualization.draw_geometries(grippers +  [scene_cloud_o3d])

    

#     end_points['batch_grasp_point'] = batch_grasp_points
#     end_points['batch_grasp_rotations'] = batch_grasp_rotations
#     end_points['batch_grasp_depth'] = batch_grasp_depth
#     end_points['batch_grasp_score'] = batch_grasp_scores
#     end_points['batch_grasp_width'] = batch_grasp_widths
#     end_points['batch_grasp_view_graspness'] = batch_view_graspness
#     end_points['batch_valid_mask'] = batch_valid_mask
#     end_points['C: Valid Points'] = valid_points_count / batch_size
#     return batch_grasp_views_rot, end_points






# def process_grasp_pseudo_label(end_points):
#     """ Process labels according to scene points and object poses. """
#     seed_xyzs = end_points['xyz_graspable']  # [B (batch size), 1024 (scene points after sample), 3]
#     pred_top_view_inds = end_points['grasp_top_view_inds']  # [B (batch size), 1024 (scene points after sample)]
#     batch_size, num_samples, _ = seed_xyzs.size()

#     valid_points_count = 0
#     valid_views_count = 0

#     batch_grasp_points = []
#     batch_grasp_views_rot = []
#     batch_view_graspness = []
#     batch_grasp_rotations = []
#     batch_grasp_depth = []
#     batch_grasp_scores = []
#     batch_grasp_widths = []
#     batch_valid_mask = []
    
    
#     for i in range(batch_size):
    
#         seed_xyz = seed_xyzs[i]  # [1024 (scene points after sample), 3]
#         ## !shsh ????
#         pred_top_view = pred_top_view_inds[i]  # [1024 (scene points after sample)]
    
#         gg_array_ema = end_points['batch_grasp_preds_ema'][i] #(N, 17)
#         grasp_top_view_inds_ema = end_points['batch_grasp_top_view_inds_ema'][i] #(N,)
#         grasp_angles_ema = end_points['batch_grasp_angles_ema'][i] #(N)
#         mat_aug = end_points['mat_aug'][i] #(N, 3, 3)
#         # grasp_q = end_points['batch_grasp_q'][i] #(N, 1)
#         view_scores_ema = end_points['batch_view_scores_ema'][i] # (N, V)

#         # points = gg_array_ema[:, 13:16]  # [N, 3]
#         # for the same points, leave only the best grasp (higher grasp_q)
#         # find the best grasp for each point
                
#         # get merged grasp points for label computation
#         # transform the view from object coordinate system to scene coordinate system
#         grasp_points_merged = []
#         grasp_views_rot_merged = []
#         grasp_rotations_merged = []
#         grasp_depth_merged = []
#         grasp_scores_merged = []
#         grasp_widths_merged = []
#         view_graspness_merged = []
#         top_view_index_merged = []
#         grasp_labels_merged = []
#         V, A, D = cfgs.num_view, 12, 4
        
#         for gg_array_t, top_view_ind, grasp_angle, view_score, mat in zip(
#             gg_array_ema, grasp_top_view_inds_ema, grasp_angles_ema, view_scores_ema, mat_aug):
            
#             grasp_score = gg_array_t[0]
#             grasp_width = gg_array_t[1]
#             grasp_depth = gg_array_t[3].unsqueeze(0)

#             pose = torch.eye(4).to(gg_array_ema.device)
#             if mat is not None:
#                 pose[:3, :3] = mat

#             # Initialize tensors for grasp data
#             grasp_scores = torch.zeros(1, V, A, D, dtype=torch.float32, device=seed_xyz.device) 
#             grasp_widths = torch.zeros(1, V, A, D, dtype=torch.float32, device=seed_xyz.device)
#             view_graspness = torch.zeros(1, V, dtype=torch.float32, device=seed_xyz.device)
#             angle_ind = (grasp_angle * A / torch.pi).long()
#             depth_ind = ((grasp_depth / 0.01) - 1).long()
            
#             grasp_points = gg_array_t[13:16].unsqueeze(0) # [1, 3]
#             grasp_scores[0, top_view_ind, angle_ind, depth_ind] = grasp_score # [1, V, A, D]
#             grasp_widths[0, top_view_ind, angle_ind, depth_ind] = grasp_width # [1, V, A, D]
#             # !TODO: shsh
#             # view_graspness[0, top_view_ind] = view_score # [1, V]
#             view_graspness = view_score.unsqueeze(0) # [1, V]

#             # select the best depth 
#             grasp_score_label_max_depth, grasp_score_label_max_depth_idx = grasp_scores.max(-1)
#             grasp_widths = grasp_widths.gather(-1, grasp_score_label_max_depth_idx.unsqueeze(-1)).squeeze(-1) # [N_good, V, A]
#             # select the best angle
#             grasp_score_label_max_angle, grasp_score_label_max_angle_idx = grasp_score_label_max_depth.max(-1)
#             grasp_depths = grasp_score_label_max_depth_idx.gather(-1, grasp_score_label_max_angle_idx.unsqueeze(-1)).squeeze(
#                 -1)
#             grasp_rotations = grasp_score_label_max_angle_idx # [N_good, V]
#             grasp_scores = grasp_score_label_max_angle # [N_good, V]
#             grasp_widths = grasp_widths.gather(-1, grasp_score_label_max_angle_idx.unsqueeze(-1)).squeeze(-1)

#             values, top_view_index = torch.topk(view_graspness, k=V)
#             grasp_rotations = torch.gather(grasp_rotations, 1, top_view_index)
#             grasp_depths = torch.gather(grasp_depths, 1, top_view_index)
#             grasp_scores = torch.gather(grasp_scores, 1, top_view_index)
#             grasp_widths = torch.gather(grasp_widths, 1, top_view_index)

#             grasp_views = generate_grasp_views(cfgs.num_view).to(pose.device)  # [300 (views), 3 (coordinate)]
#             grasp_points_trans = transform_point_cloud(grasp_points, pose, '3x4')
#             grasp_views_trans = transform_point_cloud(grasp_views, pose[:3, :3], '3x3')
#             # [300 (views), 3 (coordinate)], after translation to scene coordinate system

#             # generate and transform template grasp view rotation
#             angles = torch.zeros(grasp_views.size(0), dtype=grasp_views.dtype, device=grasp_views.device)
#             grasp_views_rot = batch_viewpoint_params_to_matrix(-grasp_views, angles)
#             grasp_views_rot_trans = torch.matmul(pose[:3, :3], grasp_views_rot)
#             # [300 (views), 3, 3 (the rotation matrix)]

#             # assign views after transform (the view will not exactly match)
#             grasp_views_ = grasp_views.transpose(0, 1).contiguous().unsqueeze(0)
#             grasp_views_trans_ = grasp_views_trans.transpose(0, 1).contiguous().unsqueeze(0)
#             view_inds = knn(grasp_views_trans_, grasp_views_, k=1).squeeze() - 1  # [300]
#             view_graspness_trans = torch.index_select(view_graspness, 1, view_inds)  # [object points, 300]
#             grasp_views_rot_trans = torch.index_select(grasp_views_rot_trans, 0, view_inds)
#             grasp_views_rot_trans = grasp_views_rot_trans.unsqueeze(0).expand(1, -1, -1, -1)
#             # [object points, 300, 3, 3]
#             # -1 means that when we transform the top 60 views into the scene coordinate,
#             # some views will have no matching
#             # It means that two views in the object coordinate match to one view in the scene coordinate
#             top_view_index_trans = (-1 * torch.ones((1, grasp_rotations.shape[1]), dtype=torch.long)
#                                     .to(seed_xyz.device))
#             tpid, tvip, tids = torch.where(view_inds == top_view_index.unsqueeze(-1))
#             top_view_index_trans[tpid, tvip] = tids  # [objects points, num_of_view]

#             # add to list
#             grasp_points_merged.append(grasp_points_trans)
#             view_graspness_merged.append(view_graspness_trans)
#             top_view_index_merged.append(top_view_index_trans)
#             grasp_rotations_merged.append(grasp_rotations)
#             grasp_depth_merged.append(grasp_depths)
#             grasp_scores_merged.append(grasp_scores)
#             grasp_widths_merged.append(grasp_widths)
#             grasp_views_rot_merged.append(grasp_views_rot_trans)

#         grasp_points_merged = torch.cat(grasp_points_merged, dim=0)  # [all object points, 3]
#         view_graspness_merged = torch.cat(view_graspness_merged, dim=0)  # [all object points, 300]
#         top_view_index_merged = torch.cat(top_view_index_merged, dim=0)  # [all object points, num_of_view]
#         grasp_rotations_merged = torch.cat(grasp_rotations_merged, dim=0).to(torch.int32)  # [all object points, num_of_view]
#         grasp_depth_merged = torch.cat(grasp_depth_merged, dim=0).to(torch.int32)  # [all object points, num_of_view]
#         grasp_scores_merged = torch.cat(grasp_scores_merged, dim=0)  # [all object points, num_of_view]
#         grasp_widths_merged = torch.cat(grasp_widths_merged, dim=0)  # [all object points, num_of_view]
#         grasp_views_rot_merged = torch.cat(grasp_views_rot_merged, dim=0)  # [all object points, 300, 3, 3]

#         # compute nearest neighbors
#         seed_xyz_ = seed_xyz.transpose(0, 1).contiguous().unsqueeze(0)
#         grasp_points_merged_ = grasp_points_merged.transpose(0, 1).contiguous().unsqueeze(0)
#         nn_inds = knn(grasp_points_merged_, seed_xyz_, k=1).squeeze() - 1

#         # assign anchor points to real points
#         grasp_points_merged = torch.index_select(grasp_points_merged, 0, nn_inds)
#         # [1024 (scene points after sample), 3]
#         grasp_views_rot_merged = torch.index_select(grasp_views_rot_merged, 0, nn_inds)
#         # [1024 (scene points after sample), 300, 3, 3]
#         view_graspness_merged = torch.index_select(view_graspness_merged, 0, nn_inds)
#         # [1024 (scene points after sample), 300]
#         top_view_index_merged = torch.index_select(top_view_index_merged, 0, nn_inds)
#         # [1024 (scene points after sample), num_of_view]
#         grasp_rotations_merged = torch.index_select(grasp_rotations_merged, 0, nn_inds)
#         # [1024 (scene points after sample), num_of_view]
#         grasp_depth_merged = torch.index_select(grasp_depth_merged, 0, nn_inds)
#         # [1024 (scene points after sample), num_of_view]
#         grasp_scores_merged = torch.index_select(grasp_scores_merged, 0, nn_inds)
#         # [1024 (scene points after sample), num_of_view]
#         grasp_widths_merged = torch.index_select(grasp_widths_merged, 0, nn_inds)
#         # [1024 (scene points after sample), num_of_view]

#         # select top view's rot, score and width
#         # we only assign labels when the pred view is in the pre-defined 60 top view, others are zero
#         pred_top_view_ = pred_top_view.view(num_samples, 1, 1, 1).expand(-1, -1, 3, 3)
#         # [1024 (points after sample), 1, 3, 3]
#         top_grasp_views_rot = torch.gather(grasp_views_rot_merged, 1, pred_top_view_).squeeze(1)
#         # [1024 (points after sample), 3, 3]
#         pid, vid = torch.where(pred_top_view.unsqueeze(-1) == top_view_index_merged)
#         # both pid and vid are [true numbers], where(condition) equals to nonzero(condition)
#         top_grasp_rotations = 12 * torch.ones(num_samples, dtype=torch.int32).to(seed_xyz.device)
#         # [1024 (points after sample)]
#         top_grasp_depth = 4 * torch.ones(num_samples, dtype=torch.int32).to(seed_xyz.device)
#         # [1024 (points after sample)]
#         top_grasp_scores = torch.zeros(num_samples, dtype=torch.float32).to(seed_xyz.device)
#         # [1024 (points after sample)]
#         top_grasp_widths = 0.1 * torch.ones(num_samples, dtype=torch.float32).to(seed_xyz.device)
#         # [1024 (points after sample)]
#         top_grasp_rotations[pid] = torch.gather(grasp_rotations_merged[pid], 1, vid.view(-1, 1)).squeeze(1)
#         top_grasp_depth[pid] = torch.gather(grasp_depth_merged[pid], 1, vid.view(-1, 1)).squeeze(1)
#         top_grasp_scores[pid] = torch.gather(grasp_scores_merged[pid], 1, vid.view(-1, 1)).squeeze(1)
#         top_grasp_widths[pid] = torch.gather(grasp_widths_merged[pid], 1, vid.view(-1, 1)).squeeze(1)

#         # only compute loss in the points with correct matching (so compute the mask first)
#         dist = compute_pointwise_dists(seed_xyz, grasp_points_merged)
#         valid_point_mask = dist < 0.005
#         valid_view_mask = torch.zeros(num_samples, dtype=torch.bool).to(seed_xyz.device)
#         valid_view_mask[pid] = True
#         valid_points_count = valid_points_count + torch.sum(valid_point_mask)
#         valid_views_count = valid_views_count + torch.sum(valid_view_mask)
#         valid_score_mask = top_grasp_scores > 0.0
#         valid_mask = valid_score_mask  & valid_view_mask  & valid_point_mask

#         # add to batch
#         batch_grasp_points.append(grasp_points_merged)
#         batch_grasp_views_rot.append(top_grasp_views_rot)
#         batch_view_graspness.append(view_graspness_merged)
#         batch_grasp_rotations.append(top_grasp_rotations)
#         batch_grasp_depth.append(top_grasp_depth)
#         batch_grasp_scores.append(top_grasp_scores)
#         batch_grasp_widths.append(top_grasp_widths)
#         batch_valid_mask.append(valid_mask)

#     batch_grasp_points = torch.stack(batch_grasp_points, 0)
#     # [B (batch size), 1024 (scene points after sample), 3]
#     batch_grasp_views_rot = torch.stack(batch_grasp_views_rot, 0)
#     # [B (batch size), 1024 (scene points after sample), 3, 3]
#     batch_view_graspness = torch.stack(batch_view_graspness, 0)
#     # [B (batch size), 1024 (scene points after sample), 300]
#     batch_grasp_rotations = torch.stack(batch_grasp_rotations, 0)
#     # [B (batch size), 1024 (scene points after sample)]
#     batch_grasp_depth = torch.stack(batch_grasp_depth, 0) # 0~4 
#     # [B (batch size), 1024 (scene points after sample)]
#     batch_grasp_scores = torch.stack(batch_grasp_scores, 0)
#     # [B (batch size), 1024 (scene points after sample)]
#     batch_grasp_widths = torch.stack(batch_grasp_widths, 0)
#     # [B (batch size), 1024 (scene points after sample)]
#     batch_valid_mask = torch.stack(batch_valid_mask, 0)
#     # [B (batch size), 1024 (scene points after sample)]
    
#     # visualize the grasps
#     import numpy as np
#     import open3d as o3d
#     from graspnetAPI.utils.utils import plot_gripper_pro_max
#     grippers = []
#     for i in range(len(batch_grasp_points[0][batch_valid_mask[0]])):

#         t = batch_grasp_points[0][batch_valid_mask[0]][i].detach().cpu().numpy()
#         R = batch_grasp_views_rot[0][batch_valid_mask[0]][i].detach().cpu().numpy()
#         width = batch_grasp_widths[0][batch_valid_mask[0]][i].detach().cpu().numpy()
#         depth = (batch_grasp_depth[0][batch_valid_mask[0]][i].detach().cpu().numpy() + 1) * 0.01
#         score = batch_grasp_scores[0][batch_valid_mask[0]][i].detach().cpu().numpy()
#         gripper = plot_gripper_pro_max(t, R, width, depth, score)
#         grippers.append(gripper)
#         # if i > 100:
#         #     break
#     scene_cloud = end_points['point_clouds_raw'][0].detach().cpu().numpy()
#     scene_cloud_o3d = o3d.geometry.PointCloud()
#     scene_cloud_o3d.points = o3d.utility.Vector3dVector(scene_cloud)
#     o3d.visualization.draw_geometries(grippers +  [scene_cloud_o3d])

    

#     end_points['batch_grasp_point'] = batch_grasp_points
#     end_points['batch_grasp_rotations'] = batch_grasp_rotations
#     end_points['batch_grasp_depth'] = batch_grasp_depth
#     end_points['batch_grasp_score'] = batch_grasp_scores
#     end_points['batch_grasp_width'] = batch_grasp_widths
#     end_points['batch_grasp_view_graspness'] = batch_view_graspness
#     end_points['batch_valid_mask'] = batch_valid_mask
#     end_points['C: Valid Points'] = valid_points_count / batch_size
#     return batch_grasp_views_rot, end_points



# def process_grasp_pseudo_label(end_points):
#     """ Process labels according to scene points and object poses. """
#     seed_xyzs = end_points['xyz_graspable']  # [B (batch size), 1024 (scene points after sample), 3]
#     pred_top_view_inds = end_points['grasp_top_view_inds']  # [B (batch size), 1024 (scene points after sample)]
#     batch_size, num_samples, _ = seed_xyzs.size()

#     valid_points_count = 0
#     valid_views_count = 0

#     batch_grasp_points = []
#     batch_grasp_views_rot = []
#     batch_view_graspness = []
#     batch_grasp_rotations = []
#     batch_grasp_depth = []
#     batch_grasp_scores = []
#     batch_grasp_widths = []
#     batch_valid_mask = []
    
    
#     for i in range(batch_size):
    
#         seed_xyz = seed_xyzs[i]  # [1024 (scene points after sample), 3]
    
#         gg_array_ema = end_points['batch_grasp_preds_ema'][i] #(N, 17)
#         grasp_top_view_inds_ema = end_points['batch_grasp_top_view_inds_ema'][i] #(N,)
#         grasp_angles_ema = end_points['batch_grasp_angles_ema'][i] #(N)
#         mat_aug = end_points['mat_aug'][i] #(N, 3, 3)
#         view_scores_ema = end_points['batch_view_scores_ema'][i] # (N, V)


                
#         # get merged grasp points for label computation
#         # transform the view from object coordinate system to scene coordinate system
#         grasp_points_merged = []
#         grasp_views_rot_merged = []
#         grasp_rotations_merged = []
#         grasp_depth_merged = []
#         grasp_scores_merged = []
#         grasp_widths_merged = []
#         view_graspness_merged = []
#         top_view_index_merged = []
#         V, A, D = cfgs.num_view, 12, 4
        
#         for gg_array_t, top_view_ind, grasp_angle, view_score, mat in zip(
#             gg_array_ema, grasp_top_view_inds_ema, grasp_angles_ema, view_scores_ema, mat_aug):
            
#             grasp_points = gg_array_t[13:16].unsqueeze(0) # [1, 3]
#             grasp_score = gg_array_t[0]
#             grasp_width = gg_array_t[1] # ~ 0.1
#             grasp_depth = gg_array_t[3]
#             grasp_depth = ((grasp_depth / 0.01) - 1).long() # 0~3
#             grasp_rotation = (grasp_angle * A / torch.pi).long() # 0~11
#             view_graspness = view_score.unsqueeze(0) # [1, V] 

#             num_grasp_points = 1
            
#             pose = torch.eye(4).to(gg_array_ema.device)
#             if mat is not None:
#                 pose[:3, :3] = mat
                
#             # generate and transform template grasp views
#             grasp_views = generate_grasp_views(cfgs.num_view).to(pose.device)  # [300 (views), 3 (coordinate)]
#             grasp_points_trans = transform_point_cloud(grasp_points, pose, '3x4')
#             grasp_views_trans = transform_point_cloud(grasp_views, pose[:3, :3], '3x3')
#             # [300 (views), 3 (coordinate)], after translation to scene coordinate system

#             # generate and transform template grasp view rotation
#             angles = torch.zeros(grasp_views.size(0), dtype=grasp_views.dtype, device=grasp_views.device)
#             grasp_views_rot = batch_viewpoint_params_to_matrix(-grasp_views, angles)
#             grasp_views_rot_trans = torch.matmul(pose[:3, :3], grasp_views_rot)

#             # assign views after transform (the view will not exactly match)
#             grasp_views_ = grasp_views.transpose(0, 1).contiguous().unsqueeze(0)
#             grasp_views_trans_ = grasp_views_trans.transpose(0, 1).contiguous().unsqueeze(0)
#             view_inds = knn(grasp_views_trans_, grasp_views_, k=1).squeeze() - 1  # [300]
            
#             # match top view index
#             top_view_index = view_inds[top_view_ind].unsqueeze(0) # [1,]
            
            
#             view_graspness_trans = torch.index_select(view_graspness, 1, view_inds)  # [object points, 300]
#             grasp_views_rot_trans = torch.index_select(grasp_views_rot_trans, 0, view_inds)
#             grasp_views_rot_trans = grasp_views_rot_trans.unsqueeze(0).expand(num_grasp_points, -1, -1, -1)
            
#             # add to list
#             grasp_points_merged.append(grasp_points_trans)
#             view_graspness_merged.append(view_graspness_trans)
#             # top_view_index_merged.append(top_view_index_trans)
#             grasp_rotations_merged.append(grasp_rotation.unsqueeze(0))
#             grasp_depth_merged.append(grasp_depth.unsqueeze(0))
#             grasp_scores_merged.append(grasp_score.unsqueeze(0))
#             grasp_widths_merged.append(grasp_width.unsqueeze(0))
#             grasp_views_rot_merged.append(grasp_views_rot_trans)
#             top_view_index_merged.append(top_view_index)

#         grasp_points_merged = torch.cat(grasp_points_merged, dim=0)  # [all object points, 3]
#         view_graspness_merged = torch.cat(view_graspness_merged, dim=0)  # [all object points, 300]
#         # top_view_index_merged = torch.cat(top_view_index_merged, dim=0)  # [all object points, num_of_view]
#         grasp_rotations_merged = torch.cat(grasp_rotations_merged, dim=0).to(torch.int32)  # [all object points]
#         grasp_depth_merged = torch.cat(grasp_depth_merged, dim=0).to(torch.int32)  # [all object points]
#         grasp_scores_merged = torch.cat(grasp_scores_merged, dim=0)  # [all object points]
#         grasp_widths_merged = torch.cat(grasp_widths_merged, dim=0)  # [all object points]
#         grasp_views_rot_merged = torch.cat(grasp_views_rot_merged, dim=0)  # [all object points, 300, 3, 3]
#         grasp_top_view_inds_ema = torch.cat(top_view_index_merged, dim=0).to(torch.int32)  # [all object points]
#         # print('grasp_points_merged', grasp_points_merged.shape)
#         # print('view_graspness_merged', view_graspness_merged.shape)
#         # print('grasp_rotations_merged', grasp_rotations_merged.shape)
#         # print('grasp_depth_merged', grasp_depth_merged.shape)
#         # print('grasp_scores_merged', grasp_scores_merged.shape)
#         # print('grasp_widths_merged', grasp_widths_merged.shape)
#         # print('grasp_views_rot_merged', grasp_views_rot_merged.shape)
#         # print('top_view_index_merged', top_view_index_merged.shape)
#         # compute nearest neighbors
#         # compute nearest neighbors to assign anchor points to scene points
#         seed_xyz_ = seed_xyz.transpose(0, 1).contiguous().unsqueeze(0)  # [1, 3, 1024]
#         grasp_points_merged_ = grasp_points_merged.transpose(0, 1).contiguous().unsqueeze(0)  # [1, 3, N_merged]
#         nn_inds = knn(grasp_points_merged_, seed_xyz_, k=1).squeeze() - 1  # [1024]

#         # assign anchor points to real points
#         grasp_points_merged = torch.index_select(grasp_points_merged, 0, nn_inds)  # [1024, 3]
#         view_graspness_merged = torch.index_select(view_graspness_merged, 0, nn_inds)  # [1024, 300]
        
#         # 단일 값들을 1024개로 확장 (각 scene point가 가장 가까운 grasp point의 값을 가짐)
#         grasp_rotations_merged = torch.index_select(grasp_rotations_merged, 0, nn_inds)  # [1024]
#         grasp_depth_merged = torch.index_select(grasp_depth_merged, 0, nn_inds)  # [1024]
#         grasp_scores_merged = torch.index_select(grasp_scores_merged, 0, nn_inds)  # [1024]
#         grasp_widths_merged = torch.index_select(grasp_widths_merged, 0, nn_inds)  # [1024]
#         grasp_views_rot_merged = torch.index_select(grasp_views_rot_merged, 0, nn_inds)  # [1024, 300, 3, 3]

#         # ===== PSEUDO GT에서는 복잡한 매칭 불필요 =====
#         # Teacher가 예측한 view index 사용 (이미 올바른 좌표계)
#         teacher_view_inds_merged = torch.index_select(grasp_top_view_inds_ema, 0, nn_inds)  # [1024] - scene point별로 할당
        
#         # Teacher 예측 기반으로 rotation matrix 선택
#         teacher_pred_view_ = teacher_view_inds_merged.view(num_samples, 1, 1, 1).expand(-1, -1, 3, 3).to(torch.int64)  # [1024, 1, 3, 3]
#         top_grasp_views_rot = torch.gather(grasp_views_rot_merged, 1, teacher_pred_view_).squeeze(1)  # [1024, 3, 3]
        
#         # Teacher 예측에 해당하는 GT 값들 직접 사용
#         # (Student 예측과 무관하게 Teacher가 선택한 view의 GT만 제공)
#         top_grasp_rotations = grasp_rotations_merged  # [1024] - 이미 teacher의 rotation
#         top_grasp_depth = grasp_depth_merged  # [1024] - 이미 teacher의 depth
#         top_grasp_scores = grasp_scores_merged  # [1024] - 이미 teacher의 score
#         top_grasp_widths = grasp_widths_merged  # [1024] - 이미 teacher의 width

#         # View graspness: Teacher의 전체 view score 사용 (어떤 view가 좋은지 정보)
#         batch_view_graspness_current = view_graspness_merged  # [1024, 300]

#         # 유효성 검증
#         # 1. 거리 기반 (pseudo GT이므로 관대하게)
#         dist = compute_pointwise_dists(seed_xyz, grasp_points_merged)  # [1024]
#         valid_point_mask = dist < 0.005  #
        
#         # 2. Score 기반 (teacher가 예측한 score가 양수인지)
#         valid_view_mask = top_grasp_scores > 0  # [1024]
        
#         valid_points_count += torch.sum(valid_point_mask)
#         valid_views_count += torch.sum(valid_view_mask)
#         valid_mask = valid_point_mask & valid_view_mask  # [1024]

#         # add to batch
#         batch_grasp_points.append(grasp_points_merged)
#         batch_grasp_views_rot.append(top_grasp_views_rot)
#         batch_view_graspness.append(batch_view_graspness_current)
#         batch_grasp_rotations.append(top_grasp_rotations)
#         batch_grasp_depth.append(top_grasp_depth)
#         batch_grasp_scores.append(top_grasp_scores)
#         batch_grasp_widths.append(top_grasp_widths)
#         batch_valid_mask.append(valid_mask)

#     batch_grasp_points = torch.stack(batch_grasp_points, 0)
#     # [B (batch size), 1024 (scene points after sample), 3]
#     batch_grasp_views_rot = torch.stack(batch_grasp_views_rot, 0)
#     # [B (batch size), 1024 (scene points after sample), 3, 3]
#     batch_view_graspness = torch.stack(batch_view_graspness, 0)
#     # [B (batch size), 1024 (scene points after sample), 300]
#     batch_grasp_rotations = torch.stack(batch_grasp_rotations, 0)
#     # [B (batch size), 1024 (scene points after sample)]
#     batch_grasp_depth = torch.stack(batch_grasp_depth, 0) # 0~4 
#     # [B (batch size), 1024 (scene points after sample)]
#     batch_grasp_scores = torch.stack(batch_grasp_scores, 0)
#     # [B (batch size), 1024 (scene points after sample)]
#     batch_grasp_widths = torch.stack(batch_grasp_widths, 0)
#     # [B (batch size), 1024 (scene points after sample)]
#     batch_valid_mask = torch.stack(batch_valid_mask, 0)
#     # [B (batch size), 1024 (scene points after sample)]
    
#     # # visualize the grasps
#     # import numpy as np
#     # import open3d as o3d
#     # from graspnetAPI.utils.utils import plot_gripper_pro_max
#     # grippers = []
#     # for i in range(len(batch_grasp_points[0][batch_valid_mask[0]])):

#     #     t = batch_grasp_points[0][batch_valid_mask[0]][i].detach().cpu().numpy()
#     #     R = batch_grasp_views_rot[0][batch_valid_mask[0]][i].detach().cpu().numpy()
#     #     width = batch_grasp_widths[0][batch_valid_mask[0]][i].detach().cpu().numpy()
#     #     depth = (batch_grasp_depth[0][batch_valid_mask[0]][i].detach().cpu().numpy() + 1) * 0.01
#     #     score = batch_grasp_scores[0][batch_valid_mask[0]][i].detach().cpu().numpy()
#     #     gripper = plot_gripper_pro_max(t, R, width, depth, score)
#     #     grippers.append(gripper)
#     #     if i > 100:
#     #         break
#     # scene_cloud = end_points['point_clouds_raw'][0].detach().cpu().numpy()
#     # scene_cloud_o3d = o3d.geometry.PointCloud()
#     # scene_cloud_o3d.points = o3d.utility.Vector3dVector(scene_cloud)
#     # o3d.visualization.draw_geometries(grippers +  [scene_cloud_o3d])

    

#     end_points['batch_grasp_point'] = batch_grasp_points
#     end_points['batch_grasp_rotations'] = batch_grasp_rotations
#     end_points['batch_grasp_depth'] = batch_grasp_depth
#     end_points['batch_grasp_score'] = batch_grasp_scores
#     end_points['batch_grasp_width'] = batch_grasp_widths
#     end_points['batch_grasp_view_graspness'] = batch_view_graspness
#     end_points['batch_valid_mask'] = batch_valid_mask
#     end_points['C: Valid Points'] = valid_points_count / batch_size
#     return batch_grasp_views_rot, end_points


import torch

# --------------------------- helpers ---------------------------

def _batched_knn_view_inds(grasp_views_trans_B, views_template_B):
    """
    배치 knn로 뷰 재매핑 인덱스 얻기.
    grasp_views_trans_B: (B, 3, V)  - 변환된 뷰들(쿼리)
    views_template_B   : (B, 3, V)  - 템플릿 뷰들(데이터베이스)
    return: view_inds (B, V) with 0-based indices
    """
    idx = knn(grasp_views_trans_B.contiguous(), views_template_B.contiguous(), k=1)
    # ext/knn 반환 형태 방어적 처리
    if idx.dim() == 3:   # (B, V, 1)
        idx = idx.squeeze(-1).squeeze(1)
    elif idx.dim() == 1: # (V,) → (1, V)
        idx = idx.unsqueeze(0)
    return (idx - 1).long()  # 원래 코드처럼 1-based → 0-based

def _unique_rot_groups(mat_aug_i, device):
    """
    mat_aug_i: 길이 N의 list/tuple (원소: None 또는 (3,3) tensor) 혹은 (N,3,3) tensor
    return: list of (key, idx_tensor, R_batch_or_None)
        - "IDENTITY": None 그룹, idx_tensor: (m,), R_batch_or_None = None
        - "ROT_g"   : 동일 회전 그룹, idx_tensor: (m,), R_batch: (m,3,3)
    """
    if torch.is_tensor(mat_aug_i):
        # 텐서로 온 경우: 모두 회전 (None 없음). 고유 회전으로 그룹핑
        R_all = mat_aug_i.to(device).contiguous()         # (N,3,3)
        N = R_all.shape[0]
        flat = R_all.reshape(N, -1)
        unique, inv = torch.unique(flat, dim=0, return_inverse=True)  # (G,9), (N,)
        groups = []
        G = unique.shape[0]
        for g in range(G):
            sel = torch.nonzero(inv == g, as_tuple=False).squeeze(1)  # (m,)
            Rg = unique[g].reshape(3,3).to(device)
            Rb = Rg.unsqueeze(0).expand(sel.numel(), -1, -1).contiguous()  # (m,3,3)
            groups.append((f"ROT_{g}", sel.long(), Rb))
        return groups

    # list/tuple: None/회전 섞여 있는 경우
    N = len(mat_aug_i)
    idx_none = [k for k, M in enumerate(mat_aug_i) if M is None]
    idx_rot  = [k for k, M in enumerate(mat_aug_i) if M is not None]

    groups = []
    if idx_none:
        groups.append(("IDENTITY", torch.tensor(idx_none, device=device, dtype=torch.long), None))

    if idx_rot:
        R_stack = torch.stack([mat_aug_i[k].to(device) for k in idx_rot], dim=0)  # (K,3,3)
        flat = R_stack.reshape(R_stack.shape[0], -1)
        unique, inv = torch.unique(flat, dim=0, return_inverse=True)              # (G,9), (K,)
        G = unique.shape[0]
        for g in range(G):
            sel_local = torch.nonzero(inv == g, as_tuple=False).squeeze(1)        # (m,)
            sel_idx   = torch.tensor([idx_rot[t.item()] for t in sel_local], device=device, dtype=torch.long)  # (m,)
            Rg = unique[g].reshape(3,3).to(device)
            Rb = Rg.unsqueeze(0).expand(sel_idx.numel(), -1, -1).contiguous()     # (m,3,3)
            groups.append((f"ROT_{g}", sel_idx, Rb))
    return groups


# --------------------------- main ---------------------------

def process_grasp_pseudo_label(end_points):
    """ Process labels according to scene points and object poses. """
    seed_xyzs = end_points['xyz_graspable']            # [B, 1024, 3]
    pred_top_view_inds = end_points['grasp_top_view_inds']  # [B, 1024] (학생 예측; 여기선 사용 X)
    batch_size, num_samples, _ = seed_xyzs.size()

    valid_points_count = 0
    valid_views_count = 0

    batch_grasp_points = []
    batch_grasp_views_rot = []
    batch_view_graspness = []
    batch_grasp_rotations = []
    batch_grasp_depth = []
    batch_grasp_scores = []
    batch_grasp_widths = []
    batch_valid_mask = []

    V, A, D = cfgs.num_view, 12, 4  # A=12 angle bins, D=4 depth bins
    device = seed_xyzs.device

    # 템플릿(고정) — 호출당 1회 생성 (원하면 전역 캐시)
    grasp_views = generate_grasp_views(V).to(device)                        # (V,3)
    angles_zero = torch.zeros(V, dtype=grasp_views.dtype, device=device)
    template_rot = batch_viewpoint_params_to_matrix(-grasp_views, angles_zero)  # (V,3,3)
    # transpose for knn 입력 형상 (B,3,V)로 쉽게 만들기 위함
    grasp_views_T = grasp_views.t().contiguous()  # (3,V)

    for i in range(batch_size):
        seed_xyz = seed_xyzs[i]  # [1024, 3]

        gg_array_ema = end_points['batch_grasp_preds_ema'][i]               # (N, 17)
        grasp_top_view_inds_ema = end_points['batch_grasp_top_view_inds_ema'][i].long()  # (N,)
        grasp_angles_ema = end_points['batch_grasp_angles_ema'][i]          # (N,)
        mat_aug = end_points['mat_aug'][i]                                   # list/tuple or tensor
        view_scores_ema = end_points['batch_view_scores_ema'][i]             # (N, V)

        N = gg_array_ema.shape[0]

        # ---- 1) 스칼라/벡터 성분 벡터화 추출 ----
        grasp_points_src = gg_array_ema[:, 13:16]                                # (N,3)
        grasp_scores_src = gg_array_ema[:, 0]                                    # (N,)
        grasp_widths_src = gg_array_ema[:, 1] / 1.2                     # (N,)
        grasp_depth_src  = ((gg_array_ema[:, 3] / 0.01) - 1).long().clamp_(0, D-1)              # (N,) 0~3
        grasp_rot_src    = (grasp_angles_ema * A / torch.pi).long().clamp_(0, A-1)              # (N,) 0~11
        view_graspness_src = view_scores_ema                                     # (N,V)

        # ---- 2) 결과 버퍼 (N 크기 단 1회 할당) ----
        grasp_points_merged    = torch.empty(N, 3,       device=device, dtype=grasp_points_src.dtype)
        view_graspness_merged  = torch.empty(N, V,       device=device, dtype=view_graspness_src.dtype)
        grasp_views_rot_merged = torch.empty(N, V, 3, 3, device=device, dtype=template_rot.dtype)
        grasp_top_view_inds_vec= torch.empty(N,          device=device, dtype=torch.long)

        # ---- 3) 고유 회전 그룹별 계산 (뷰 knn 포함) ----
        groups = _unique_rot_groups(mat_aug, device)

        for key, idxs, R_batch in groups:
            m = idxs.numel()
            if key == "IDENTITY":
                # p' = p
                grasp_points_merged[idxs] = grasp_points_src[idxs]
                # 뷰: 변환 없음 → 변환된 뷰 = 템플릿
                # knn은 유지: 배치로 동일 템플릿/템플릿 매칭
                grasp_views_trans_B = grasp_views_T.unsqueeze(0).expand(m, -1, -1).contiguous()  # (m,3,V)
                grasp_views_B       = grasp_views_T.unsqueeze(0).expand(m, -1, -1).contiguous()  # (m,3,V)
                view_inds = _batched_knn_view_inds(grasp_views_trans_B, grasp_views_B)          # (m,V)
                # view score 재정렬
                vg = torch.gather(view_graspness_src[idxs], 1, view_inds)                       # (m,V)
                view_graspness_merged[idxs] = vg
                # 회전행렬 템플릿 재정렬
                tr = template_rot.unsqueeze(0).expand(m, -1, -1, -1)                             # (m,V,3,3)
                gr = tr.gather(1, view_inds.view(m, V, 1, 1).expand(m, V, 3, 3))                 # (m,V,3,3)
                grasp_views_rot_merged[idxs] = gr
                # top-view 인덱스 재매핑
                grasp_top_view_inds_vec[idxs] = view_inds[
                    torch.arange(m, device=device), grasp_top_view_inds_ema[idxs]
                ]

            else:
                # R_batch: (m,3,3), 같은 회전 반복일 수도 있고(grouping으로 m>1)
                # 포인트 회전: p' = R p
                gp = torch.bmm(R_batch, grasp_points_src[idxs].unsqueeze(-1)).squeeze(-1)        # (m,3)
                grasp_points_merged[idxs] = gp

                # 뷰 회전: v' = R v  → (m,3,V)
                v_trans = torch.bmm(R_batch, grasp_views_T.unsqueeze(0).expand(m, -1, -1))       # (m,3,V)

                # 뷰 매핑 knn (배치 1회)
                grasp_views_B = grasp_views_T.unsqueeze(0).expand(m, -1, -1).contiguous()        # (m,3,V)
                view_inds = _batched_knn_view_inds(v_trans, grasp_views_B)                       # (m,V)

                # view score 재정렬
                vg = torch.gather(view_graspness_src[idxs], 1, view_inds)                        # (m,V)
                view_graspness_merged[idxs] = vg

                # 회전행렬 템플릿 좌승: (m,V,3,3)
                tr = template_rot.unsqueeze(0).expand(m, -1, -1, -1)                             # (m,V,3,3)
                # (m,1,3,3) @ (m,V,3,3) → (m,V,3,3)
                gr = torch.matmul(R_batch.unsqueeze(1), tr)                                      # (m,V,3,3)
                # V축 재정렬(view_inds)
                gr = gr.gather(1, view_inds.view(m, V, 1, 1).expand(m, V, 3, 3))                 # (m,V,3,3)
                grasp_views_rot_merged[idxs] = gr

                # top-view 인덱스 재매핑
                grasp_top_view_inds_vec[idxs] = view_inds[
                    torch.arange(m, device=device), grasp_top_view_inds_ema[idxs]
                ]

        # ---- 4) 씬 포인트(1024)로 최근접 할당 (원래 knn 그대로) ----
        seed_xyz_ = seed_xyz.transpose(0, 1).contiguous().unsqueeze(0)                 # (1,3,1024)
        grasp_points_merged_ = grasp_points_merged.transpose(0, 1).contiguous().unsqueeze(0)  # (1,3,N)
        nn_inds = knn(grasp_points_merged_, seed_xyz_, k=1).squeeze() - 1              # (1024)

        # 1024개로 인덱싱
        grasp_points_merged = torch.index_select(grasp_points_merged, 0, nn_inds)      # (1024,3)
        view_graspness_merged = torch.index_select(view_graspness_merged, 0, nn_inds)  # (1024,V)
        grasp_rotations_merged = torch.index_select(grasp_rot_src.to(torch.int32), 0, nn_inds) # (1024,)
        grasp_depth_merged    = torch.index_select(grasp_depth_src.to(torch.int32),    0, nn_inds) # (1024,)
        grasp_scores_merged   = torch.index_select(grasp_scores_src, 0, nn_inds)       # (1024,)
        grasp_widths_merged   = torch.index_select(grasp_widths_src, 0, nn_inds)       # (1024,)
        grasp_views_rot_merged= torch.index_select(grasp_views_rot_merged, 0, nn_inds) # (1024,V,3,3)
        grasp_top_view_inds_ema = torch.index_select(grasp_top_view_inds_vec, 0, nn_inds).long()  # (1024,)

        # view_graspness_merged = torch.zeros((num_samples, V), device=device, dtype=view_graspness_src.dtype)  # (1024,V)
        # view_graspness_merged[grasp_top_view_inds_ema] = 1.0
        # # torch.index_select(view_graspness_merged, 0, nn_inds)  # (1024,V)
        # view_graspness_merged = torch.zeros((num_samples, V), device=device, dtype=view_graspness_src.dtype)
        
        # make smooth the view_graspness with temperature then set the top view to 1
        view_graspness_merged = torch.clamp(view_graspness_merged, min=1e-6)
        view_graspness_merged = view_graspness_merged / 0.1
        view_graspness_merged = torch.softmax(view_graspness_merged , dim=1)
        ar = torch.arange(num_samples, device=device)
        view_graspness_merged[ar, grasp_top_view_inds_ema] = 1.0
        
        # visualize view_graspness (x-axis: view index, y-axis: graspness)
        # import matplotlib.pyplot as plt
        # plt.imshow(view_graspness_merged.cpu().numpy(), cmap='hot', interpolation='nearest')
        # plt.colorbar()
        # plt.title('View Graspness Heatmap')
        # plt.xlabel('View Index')
        # plt.ylabel('Point Index')
        # plt.show()
        # plt.savefig('view_graspness_heatmap.png')


        # ---- 5) teacher view로 최종 회전행렬 선택 (원래 방식) ----
        teacher_pred_view_ = grasp_top_view_inds_ema.view(-1, 1, 1, 1).expand(-1, 1, 3, 3)  # (1024,1,3,3)
        top_grasp_views_rot = torch.gather(grasp_views_rot_merged, 1, teacher_pred_view_).squeeze(1)  # (1024,3,3)

        # ---- 6) 유효 마스크 ----
        dist = compute_pointwise_dists(seed_xyz, grasp_points_merged)  # (1024,)
        valid_point_mask = dist < 0.005
        valid_view_mask  = grasp_scores_merged > 0
        valid_mask       = valid_point_mask & valid_view_mask

        valid_points_count += torch.sum(valid_point_mask)
        valid_views_count  += torch.sum(valid_view_mask)

        # ---- 7) 배치 누적 ----
        batch_grasp_points.append(grasp_points_merged)
        batch_grasp_views_rot.append(top_grasp_views_rot)
        batch_view_graspness.append(view_graspness_merged)
        batch_grasp_rotations.append(grasp_rotations_merged)
        batch_grasp_depth.append(grasp_depth_merged)
        batch_grasp_scores.append(grasp_scores_merged)
        batch_grasp_widths.append(grasp_widths_merged)
        batch_valid_mask.append(valid_mask)

    # ---- 8) 배치 스택 & 저장 ----
    batch_grasp_points     = torch.stack(batch_grasp_points, 0)       # [B, 1024, 3]
    batch_grasp_views_rot  = torch.stack(batch_grasp_views_rot, 0)    # [B, 1024, 3, 3]
    batch_view_graspness   = torch.stack(batch_view_graspness, 0)     # [B, 1024, V]
    batch_grasp_rotations  = torch.stack(batch_grasp_rotations, 0)    # [B, 1024]
    batch_grasp_depth      = torch.stack(batch_grasp_depth, 0)        # [B, 1024]
    batch_grasp_scores     = torch.stack(batch_grasp_scores, 0)       # [B, 1024]
    batch_grasp_widths     = torch.stack(batch_grasp_widths, 0)       # [B, 1024]
    batch_valid_mask       = torch.stack(batch_valid_mask, 0)         # [B, 1024]


    # # visualize the grasps
    # import numpy as np
    # import open3d as o3d
    # from graspnetAPI.utils.utils import plot_gripper_pro_max
    # print("Valid grasps:", batch_valid_mask[0].sum().item())
    # grippers = []
    # for i in range(len(batch_grasp_points[0][batch_valid_mask[0]])):

    #     t = batch_grasp_points[0][batch_valid_mask[0]][i].detach().cpu().numpy()
    #     R = batch_grasp_views_rot[0][batch_valid_mask[0]][i].detach().cpu().numpy()
    #     width = batch_grasp_widths[0][batch_valid_mask[0]][i].detach().cpu().numpy()
    #     depth = (batch_grasp_depth[0][batch_valid_mask[0]][i].detach().cpu().numpy() + 1) * 0.01
    #     score = batch_grasp_scores[0][batch_valid_mask[0]][i].detach().cpu().numpy()
    #     gripper = plot_gripper_pro_max(t, R, width, depth, score)
    #     grippers.append(gripper)
        # if i > 100:
        #     break
    # scene_cloud = end_points['point_clouds_raw'][0].detach().cpu().numpy()
    # scene_cloud_o3d = o3d.geometry.PointCloud()
    # scene_cloud_o3d.points = o3d.utility.Vector3dVector(scene_cloud)
    # o3d.visualization.draw_geometries(grippers +  [scene_cloud_o3d])


    end_points['batch_grasp_point'] = batch_grasp_points
    end_points['batch_grasp_rotations'] = batch_grasp_rotations
    end_points['batch_grasp_depth'] = batch_grasp_depth
    end_points['batch_grasp_score'] = batch_grasp_scores
    end_points['batch_grasp_width'] = batch_grasp_widths
    end_points['batch_grasp_view_graspness'] = batch_view_graspness
    end_points['batch_valid_mask'] = batch_valid_mask
    end_points['C: Valid Points'] = valid_points_count / batch_size

    return batch_grasp_views_rot, end_points




    



def process_grasp_labels(end_points):
    """ Process labels according to scene points and object poses. """
    seed_xyzs = end_points['xyz_graspable']  # [B (batch size), 1024 (scene points after sample), 3]
    pred_top_view_inds = end_points['grasp_top_view_inds']  # [B (batch size), 1024 (scene points after sample)]
    batch_size, num_samples, _ = seed_xyzs.size()

    valid_points_count = 0
    valid_views_count = 0

    batch_grasp_points = []
    batch_grasp_views_rot = []
    batch_view_graspness = []
    batch_grasp_rotations = []
    batch_grasp_depth = []
    batch_grasp_scores = []
    batch_grasp_widths = []
    batch_valid_mask = []
    for i in range(batch_size):
        seed_xyz = seed_xyzs[i]  # [1024 (scene points after sample), 3]
        pred_top_view = pred_top_view_inds[i]  # [1024 (scene points after sample)]
        poses = end_points['object_poses_list'][i]  # a list with length of object amount, each has size [3, 4]

        # get merged grasp points for label computation
        # transform the view from object coordinate system to scene coordinate system
        grasp_points_merged = []
        grasp_views_rot_merged = []
        grasp_rotations_merged = []
        grasp_depth_merged = []
        grasp_scores_merged = []
        grasp_widths_merged = []
        view_graspness_merged = []
        top_view_index_merged = []
        for obj_idx, pose in enumerate(poses):
            grasp_points = end_points['grasp_points_list'][i][obj_idx]  # [objects points, 3]
            grasp_rotations = end_points['grasp_rotations_list'][i][obj_idx]  # [objects points, num_of_view]
            grasp_depth = end_points['grasp_depth_list'][i][obj_idx]  # [objects points, num_of_view]
            grasp_scores = end_points['grasp_scores_list'][i][obj_idx]  # [objects points, num_of_view]
            grasp_widths = end_points['grasp_widths_list'][i][obj_idx]  # [objects points, num_of_view]
            view_graspness = end_points['view_graspness_list'][i][obj_idx]  # [objects points, 300]
            top_view_index = end_points['top_view_index_list'][i][obj_idx]  # [objects points, num_of_view]
            num_grasp_points = grasp_points.size(0)
            
            # generate and transform template grasp views
            grasp_views = generate_grasp_views(cfgs.num_view).to(pose.device)  # [300 (views), 3 (coordinate)]
            grasp_points_trans = transform_point_cloud(grasp_points, pose, '3x4')
            grasp_views_trans = transform_point_cloud(grasp_views, pose[:3, :3], '3x3')
            # [300 (views), 3 (coordinate)], after translation to scene coordinate system

            # generate and transform template grasp view rotation
            angles = torch.zeros(grasp_views.size(0), dtype=grasp_views.dtype, device=grasp_views.device)
            grasp_views_rot = batch_viewpoint_params_to_matrix(-grasp_views, angles)
            grasp_views_rot_trans = torch.matmul(pose[:3, :3], grasp_views_rot)
            # [300 (views), 3, 3 (the rotation matrix)]

            # assign views after transform (the view will not exactly match)
            grasp_views_ = grasp_views.transpose(0, 1).contiguous().unsqueeze(0)
            grasp_views_trans_ = grasp_views_trans.transpose(0, 1).contiguous().unsqueeze(0)
            view_inds = knn(grasp_views_trans_, grasp_views_, k=1).squeeze() - 1  # [300]
            view_graspness_trans = torch.index_select(view_graspness, 1, view_inds)  # [object points, 300]
            grasp_views_rot_trans = torch.index_select(grasp_views_rot_trans, 0, view_inds)
            grasp_views_rot_trans = grasp_views_rot_trans.unsqueeze(0).expand(num_grasp_points, -1, -1, -1)
            # [object points, 300, 3, 3]
            # -1 means that when we transform the top 60 views into the scene coordinate,
            # some views will have no matching
            # It means that two views in the object coordinate match to one view in the scene coordinate
            top_view_index_trans = (-1 * torch.ones((num_grasp_points, grasp_rotations.shape[1]), dtype=torch.long)
                                    .to(seed_xyz.device))
            tpid, tvip, tids = torch.where(view_inds == top_view_index.unsqueeze(-1))
            top_view_index_trans[tpid, tvip] = tids  # [objects points, num_of_view]

            # add to list
            grasp_points_merged.append(grasp_points_trans)
            view_graspness_merged.append(view_graspness_trans)
            top_view_index_merged.append(top_view_index_trans)
            grasp_rotations_merged.append(grasp_rotations)
            grasp_depth_merged.append(grasp_depth)
            grasp_scores_merged.append(grasp_scores)
            grasp_widths_merged.append(grasp_widths)
            grasp_views_rot_merged.append(grasp_views_rot_trans)

        grasp_points_merged = torch.cat(grasp_points_merged, dim=0)  # [all object points, 3]
        view_graspness_merged = torch.cat(view_graspness_merged, dim=0)  # [all object points, 300]
        top_view_index_merged = torch.cat(top_view_index_merged, dim=0)  # [all object points, num_of_view]
        grasp_rotations_merged = torch.cat(grasp_rotations_merged, dim=0)  # [all object points, num_of_view]
        grasp_depth_merged = torch.cat(grasp_depth_merged, dim=0)  # [all object points, num_of_view]
        grasp_scores_merged = torch.cat(grasp_scores_merged, dim=0)  # [all object points, num_of_view]
        grasp_widths_merged = torch.cat(grasp_widths_merged, dim=0)  # [all object points, num_of_view]
        grasp_views_rot_merged = torch.cat(grasp_views_rot_merged, dim=0)  # [all object points, 300, 3, 3]

        # compute nearest neighbors
        seed_xyz_ = seed_xyz.transpose(0, 1).contiguous().unsqueeze(0)
        grasp_points_merged_ = grasp_points_merged.transpose(0, 1).contiguous().unsqueeze(0)
        nn_inds = knn(grasp_points_merged_, seed_xyz_, k=1).squeeze() - 1

        # assign anchor points to real points
        grasp_points_merged = torch.index_select(grasp_points_merged, 0, nn_inds)
        # [1024 (scene points after sample), 3]
        grasp_views_rot_merged = torch.index_select(grasp_views_rot_merged, 0, nn_inds)
        # [1024 (scene points after sample), 300, 3, 3]
        view_graspness_merged = torch.index_select(view_graspness_merged, 0, nn_inds)
        # [1024 (scene points after sample), 300]
        top_view_index_merged = torch.index_select(top_view_index_merged, 0, nn_inds)
        # [1024 (scene points after sample), num_of_view]
        grasp_rotations_merged = torch.index_select(grasp_rotations_merged, 0, nn_inds)
        # [1024 (scene points after sample), num_of_view]
        grasp_depth_merged = torch.index_select(grasp_depth_merged, 0, nn_inds)
        # [1024 (scene points after sample), num_of_view]
        grasp_scores_merged = torch.index_select(grasp_scores_merged, 0, nn_inds)
        # [1024 (scene points after sample), num_of_view]
        grasp_widths_merged = torch.index_select(grasp_widths_merged, 0, nn_inds)
        # [1024 (scene points after sample), num_of_view]

        # select top view's rot, score and width
        # we only assign labels when the pred view is in the pre-defined 60 top view, others are zero
        pred_top_view_ = pred_top_view.view(num_samples, 1, 1, 1).expand(-1, -1, 3, 3)
        # [1024 (points after sample), 1, 3, 3]
        top_grasp_views_rot = torch.gather(grasp_views_rot_merged, 1, pred_top_view_).squeeze(1)
        # [1024 (points after sample), 3, 3]
        pid, vid = torch.where(pred_top_view.unsqueeze(-1) == top_view_index_merged)
        # both pid and vid are [true numbers], where(condition) equals to nonzero(condition)
        top_grasp_rotations = 12 * torch.ones(num_samples, dtype=torch.int32).to(seed_xyz.device)
        # [1024 (points after sample)]
        top_grasp_depth = 4 * torch.ones(num_samples, dtype=torch.int32).to(seed_xyz.device)
        # [1024 (points after sample)]
        top_grasp_scores = torch.zeros(num_samples, dtype=torch.float32).to(seed_xyz.device)
        # [1024 (points after sample)]
        top_grasp_widths = 0.1 * torch.ones(num_samples, dtype=torch.float32).to(seed_xyz.device)
        # [1024 (points after sample)]
        top_grasp_rotations[pid] = torch.gather(grasp_rotations_merged[pid], 1, vid.view(-1, 1)).squeeze(1)
        top_grasp_depth[pid] = torch.gather(grasp_depth_merged[pid], 1, vid.view(-1, 1)).squeeze(1)
        top_grasp_scores[pid] = torch.gather(grasp_scores_merged[pid], 1, vid.view(-1, 1)).squeeze(1)
        top_grasp_widths[pid] = torch.gather(grasp_widths_merged[pid], 1, vid.view(-1, 1)).squeeze(1)

        # only compute loss in the points with correct matching (so compute the mask first)
        dist = compute_pointwise_dists(seed_xyz, grasp_points_merged)
        valid_point_mask = dist < 0.005
        valid_view_mask = torch.zeros(num_samples, dtype=torch.bool).to(seed_xyz.device)
        valid_view_mask[pid] = True
        valid_points_count = valid_points_count + torch.sum(valid_point_mask)
        valid_views_count = valid_views_count + torch.sum(valid_view_mask)
        valid_mask = valid_point_mask & valid_view_mask

        # add to batch
        batch_grasp_points.append(grasp_points_merged)
        batch_grasp_views_rot.append(top_grasp_views_rot)
        batch_view_graspness.append(view_graspness_merged)
        batch_grasp_rotations.append(top_grasp_rotations)
        batch_grasp_depth.append(top_grasp_depth)
        batch_grasp_scores.append(top_grasp_scores)
        batch_grasp_widths.append(top_grasp_widths)
        batch_valid_mask.append(valid_mask)

    batch_grasp_points = torch.stack(batch_grasp_points, 0)
    # [B (batch size), 1024 (scene points after sample), 3]
    batch_grasp_views_rot = torch.stack(batch_grasp_views_rot, 0)
    # [B (batch size), 1024 (scene points after sample), 3, 3]
    batch_view_graspness = torch.stack(batch_view_graspness, 0)
    # [B (batch size), 1024 (scene points after sample), 300]
    batch_grasp_rotations = torch.stack(batch_grasp_rotations, 0)
    # [B (batch size), 1024 (scene points after sample)]
    batch_grasp_depth = torch.stack(batch_grasp_depth, 0)
    # [B (batch size), 1024 (scene points after sample)]
    batch_grasp_scores = torch.stack(batch_grasp_scores, 0)
    # [B (batch size), 1024 (scene points after sample)]
    batch_grasp_widths = torch.stack(batch_grasp_widths, 0)
    # [B (batch size), 1024 (scene points after sample)]
    batch_valid_mask = torch.stack(batch_valid_mask, 0)
    # [B (batch size), 1024 (scene points after sample)]



    end_points['batch_grasp_point'] = batch_grasp_points
    end_points['batch_grasp_rotations'] = batch_grasp_rotations
    end_points['batch_grasp_depth'] = batch_grasp_depth
    end_points['batch_grasp_score'] = batch_grasp_scores
    end_points['batch_grasp_width'] = batch_grasp_widths
    end_points['batch_grasp_view_graspness'] = batch_view_graspness
    end_points['batch_valid_mask'] = batch_valid_mask
    end_points['C: Valid Points'] = valid_points_count / batch_size
    return batch_grasp_views_rot, end_points