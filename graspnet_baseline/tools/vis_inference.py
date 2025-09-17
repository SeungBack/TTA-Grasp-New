import numpy as np
import os 
import json
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from graspnetAPI import GraspGroup, GraspNetEval

import os
import sys
import numpy as np
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

import open3d as o3d
import wandb  # Import wandb

from omegaconf import OmegaConf

# Add paths to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, '../dataset'))
from graspnet_dataset import GraspNetDataset, collate_fn


# split = 'test_similar_mixed_mini'
split = 'test_similar_mixed'
dump_folder = '/home/seung/Workspaces/grasp/TestAdaGrasp/GraspNet-Baseline/logs'
root = '/home/seung/Workspaces/Datasets/GraspNet-1Billion'

with open(os.path.join(root, 'splits', '{}.json'.format(split)), 'r') as f:
    scene_id_img_id_pairs = json.load(f)

test_dataset = GraspNetDataset(
    root,
    valid_obj_idxs=None, 
    grasp_labels=None, 
    split=split,
    camera='realsense', 
    num_points=20000, 
    remove_outlier=True, 
    augment=False, 
    load_label=False, 
    return_raw_cloud=True,
)

print(f'Test dataset size: {len(test_dataset)}')



method = 'graspnet1b/notta/realsense_similar'
# method= 'graspnet1b/tta-grasp/exp_bs1/realsense_similar_hflip_nolamda'

for scene_id, img_id in tqdm(scene_id_img_id_pairs[1000:1010]):
    gg_array_path = os.path.join(dump_folder, method, 'scene_{:04d}'.format(scene_id), 'realsense', '{:04d}.npy'.format(img_id))
    score_path = os.path.join(dump_folder, method, 'scene_{:04d}'.format(scene_id), 'realsense', '{:04d}_acc.npy'.format(img_id))
    grasp_group_array = np.load(gg_array_path)
    scores = np.load(score_path)
    scores = scores.mean(axis=1)

    gg = GraspGroup()
    gg.grasp_group_array = grasp_group_array
    gg.sort_by_score()
    gg.grasp_group_array = gg.grasp_group_array[:50]
    gg.grasp_group_array[:, 0] = scores

    gg_o3d = gg.to_open3d_geometry_list()
    data = test_dataset[60]
    cloud = data['point_clouds_raw']
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud)
    o3d.visualization.draw_geometries([cloud_o3d, *gg_o3d])
