""" GraspNet dataset processing.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import scipy.io as scio
from PIL import Image
import json
import cv2

import torch
try:
    from torch._six import container_abcs
except ImportError:
    import collections.abc as container_abcs
from torch.utils.data import Dataset
from tqdm import tqdm
import random

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, '../utils'))
from data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image,\
                            get_workspace_mask_gc6d, remove_invisible_grasp_points

def sample_points(cloud, colors=None, num_points=20000):
    if len(cloud) >= num_points:
        idxs = np.random.choice(len(cloud), num_points, replace=False)
    else:
        idxs1 = np.arange(len(cloud))
        idxs2 = np.random.choice(len(cloud), num_points-len(cloud), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    if colors is not None:
        return cloud[idxs], colors[idxs]
    else:
        return cloud[idxs]

class GraspClutter6DDataset(Dataset):
    def __init__(self, root, valid_obj_idxs, grasp_labels, camera='azure-kinect', split='train', num_points=20000,
                 remove_outlier=False, remove_invisible=True, augment=False, load_label=True, return_raw_cloud=False):
        assert(num_points<=50000)
        self.root = root
        self.split = split
        self.num_points = num_points
        self.remove_outlier = remove_outlier
        self.remove_invisible = remove_invisible
        self.valid_obj_idxs = valid_obj_idxs
        self.grasp_labels = grasp_labels
        self.camera = camera
        self.augment = augment
        self.load_label = load_label
        self.collision_labels = {}

        if 'all' in split:
            with open(os.path.join(self.root, 'split_info', 'grasp_train_scene_ids.json')) as f:
                self.sceneIds = [int(x) for x in json.load(f)]
            with open(os.path.join(self.root, 'split_info', 'grasp_test_scene_ids.json')) as f:
                self.sceneIds += [int(x) for x in json.load(f)]
        elif 'train' in split:
            with open(os.path.join(self.root, 'split_info', 'grasp_train_scene_ids.json')) as f:
                self.sceneIds = [int(x) for x in json.load(f)]
        elif 'test' in split:
            with open(os.path.join(self.root, 'split_info', 'grasp_test_scene_ids.json')) as f:
                self.sceneIds = [int(x) for x in json.load(f)]
        else:
            raise ValueError('Invalid split: {} (must be train/test/all)'.format(split))
        self.sceneIds = ['{}'.format(str(x).zfill(6)) for x in self.sceneIds]
        if self.camera == 'kinect':
            self.camera = 'azure-kinect' # for GraspNet-1Billon compatibility
        if self.camera == 'realsense':
            self.camera = 'realsense-d435' # for GraspNet-1Billon compatibility
        self.index_to_info = {}
        index = 0
        self.scene_ids = []
        for sceneid in tqdm(self.sceneIds, desc='Loading data path and collision labels...'):
            if 'mini' in self.split:
                ann_ids = random.sample(range(13), 1)
            else:
                ann_ids = range(13)
            for ann_id in ann_ids:
                img_num = 4*ann_id
                if self.camera == 'realsense-d415':
                    img_num += 1
                    self.factor_depth = 1000.0
                elif self.camera == 'realsense-d435':
                    img_num += 2
                    self.factor_depth = 1000.0
                elif self.camera == 'azure-kinect':
                    img_num += 3
                    self.factor_depth = 10000.0
                elif self.camera == 'zivid':
                    img_num += 4
                    self.factor_depth = 10000.0
                else:
                    raise ValueError('Invalid camera type')
                
                if sceneid == '000401' and img_num == 7:
                    continue
                if sceneid == '000404' and img_num == 7:
                    continue
                self.index_to_info[index] = (sceneid, img_num)
                index += 1
                self.scene_ids.append(sceneid)
        # random shuffle self.index_to_info
        random.shuffle(self.index_to_info)
        print('Total number of images for {}: {}'.format(split, len(self.scene_ids)))
        self.return_raw_cloud = return_raw_cloud
        self.frameid = [x[1] for x in self.index_to_info.values()]

    def scene_list(self):
        return self.scene_ids

    def __len__(self):
        return len(self.index_to_info)

    def augment_data(self, point_clouds, object_poses_list):
        # Flipping along the YZ plane
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 1]])
            point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)

        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random()*np.pi/3) - np.pi/6 # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0],
                            [0, c,-s],
                            [0, s, c]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        for i in range(len(object_poses_list)):
            object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)
        return point_clouds, object_poses_list

    def __getitem__(self, index):
        if self.load_label:
            return self.get_data_label(index)
        else:
            return self.get_data(index)

    def get_data(self, index, return_raw_cloud=False):
        
        sceneid, img_num = self.index_to_info[index]
        color_path = os.path.join(self.root, 'scenes', str(sceneid).zfill(6), 'rgb', str(img_num).zfill(6)+'.png')
        depth_path = os.path.join(self.root, 'scenes', str(sceneid).zfill(6), 'depth', str(img_num).zfill(6)+'.png')
        label_path = os.path.join(self.root, 'scenes', str(sceneid).zfill(6), 'label', str(img_num).zfill(6)+'.png')

        color = np.array(Image.open(color_path), dtype=np.float32) / 255.0
        depth = np.array(cv2.imread(depth_path, cv2.IMREAD_UNCHANGED))
        seg = np.array(cv2.imread(label_path))[:, :, 0]
        scene_camera_path = os.path.join(self.root, 'scenes', str(sceneid).zfill(6), 'scene_camera.json')
        with open(scene_camera_path) as f:
            scene_camera = json.load(f)
        H, W = depth.shape
        intrinsic = np.array(scene_camera[str(img_num)]['cam_K']).reshape((3,3))
        camera = CameraInfo(W, H, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], self.factor_depth)
        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
        cloud = np.ascontiguousarray(cloud, dtype=np.float32)

        # get valid points
        depth_mask = (depth > 0)
       
        if self.remove_outlier:
            cam_R_w2c = np.array(scene_camera[str(img_num)]['cam_R_w2c']).reshape((3,3))
            cam_t_w2c = np.array(scene_camera[str(img_num)]['cam_t_w2c']).reshape((3,1))
            camera_pose = np.eye(4)
            camera_pose[:3,:3] = cam_R_w2c
            camera_pose[:3,3] = cam_t_w2c.squeeze() / 1000.0
            workspace_mask = get_workspace_mask_gc6d(cloud, seg, trans=camera_pose, organized=True, outlier=0.1)
            mask = depth_mask & workspace_mask
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        
        cloud_sampled, color_sampled = sample_points(cloud_masked, color_masked, self.num_points)
        ret_dict = {}
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['cloud_colors'] = color_sampled.astype(np.float32)
        if self.return_raw_cloud or return_raw_cloud:
            ret_dict['point_clouds_raw'] = sample_points(cloud_masked, num_points=100000).astype(np.float32)
        return ret_dict

    def get_data_label(self, index):

        sceneid, img_num = self.index_to_info[index]
        color_path = os.path.join(self.root, 'real', str(sceneid).zfill(6), 'rgb', str(img_num).zfill(6)+'.png')
        label_path = os.path.join(self.root, 'real', str(sceneid).zfill(6), 'label', str(img_num).zfill(6)+'.png')
        depth_path = os.path.join(self.root, 'real', str(sceneid).zfill(6), 'depth', str(img_num).zfill(6)+'.png')
        collision_labels = np.load(os.path.join(self.root, 'collision_label', str(sceneid).zfill(6) + '.npz'))

        color = np.array(Image.open(color_path), dtype=np.float32) / 255.0
        depth = np.array(cv2.imread(depth_path, cv2.IMREAD_UNCHANGED))
        seg = np.array(cv2.imread(label_path))[:, :, 0]

        scene_camera_path = os.path.join(self.root, 'real', str(sceneid).zfill(6), 'scene_camera.json')
        scene_gt_path = os.path.join(self.root, 'real', str(sceneid).zfill(6), 'scene_gt.json')
        with open(scene_camera_path) as f:
            scene_camera = json.load(f)
        with open(scene_gt_path) as f:
            scene_gt = json.load(f)
        H, W = depth.shape
        intrinsic = np.array(scene_camera[str(img_num)]['cam_K']).reshape((3,3))
        camera = CameraInfo(W, H, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], self.factor_depth)
        obj_list = []
        for obj in scene_gt[str(img_num)]:
            obj_list.append(obj['obj_id'])
        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
        cloud = np.ascontiguousarray(cloud, dtype=np.float32)


        depth_mask = (depth > 0)

        if self.remove_outlier:
            cam_R_w2c = np.array(scene_camera[str(img_num)]['cam_R_w2c']).reshape((3,3))
            cam_t_w2c = np.array(scene_camera[str(img_num)]['cam_t_w2c']).reshape((3,1))
            camera_pose = np.eye(4)
            camera_pose[:3,:3] = cam_R_w2c
            camera_pose[:3,3] = cam_t_w2c.squeeze() / 1000.0
            workspace_mask = get_workspace_mask_gc6d(cloud, seg, trans=camera_pose, organized=True, outlier=0.1)
            mask = depth_mask & workspace_mask
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = seg[mask]
        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        seg_sampled = seg_masked[idxs]

        objectness_label = seg_sampled.copy()
        objectness_label[objectness_label>1] = 1
        
        object_poses_list = []
        grasp_points_list = []
        grasp_offsets_list = []
        grasp_scores_list = []
        grasp_tolerance_list = []
        grasp_collision_list = []

        for i, obj_idx in enumerate(obj_list):

            if obj_idx not in self.valid_obj_idxs:
                continue
            if (seg_sampled == obj_idx).sum() < 50:
                continue
            pose = np.eye(4)[:3, :]
            pose[:3,:3] = np.array(obj['cam_R_m2c']).reshape((3,3))
            pose[:3,3] = np.array(obj['cam_t_m2c']).reshape((3,)) / 1000.0
            pose = pose.astype(np.float32)
            object_poses_list.append(pose)
            points, offsets, scores, tolerance = self.grasp_labels[obj_idx]

            if len(points) > 4096:
                # first downsample to 4096
                idxs = np.random.choice(len(points), 4096, replace=False)
                points = points[idxs]
                offsets = offsets[idxs]
                scores = scores[idxs]
                tolerance = tolerance[idxs]
                collision = collision[idxs]

            collision = collision_labels['arr_{}'.format(i)].copy()
            # remove invisible grasp points
            if self.remove_invisible:
                visible_mask = remove_invisible_grasp_points(cloud_sampled[seg_sampled==obj_idx], points, pose, th=0.05)
                points = points[visible_mask]
                offsets = offsets[visible_mask]
                scores = scores[visible_mask]
                tolerance = tolerance[visible_mask]
                collision = collision[visible_mask]

            # max_len = 2048
            # if len(points) > max_len:
            #     idxs = np.random.choice(len(points), max_len, replace=False)
            #     points = points[idxs]
            #     offsets = offsets[idxs]
            #     scores = scores[idxs]
            #     tolerance = tolerance[idxs]
            #     collision = collision[idxs]

            _scores = scores.copy()
            _scores[collision] = 0
            _scores[_scores < 0] = 0
            _scores[_scores > 0] = 1.1 - _scores[_scores > 0]
            n_sample_points = min(int(len(points)), 350)
            scores_sum = _scores.sum(axis=(1, 2, 3))
            sorted_indices = np.argsort(-scores_sum)
            non_collision_idxs = sorted_indices[:int((scores_sum>0).sum())]

            all_indices = np.arange(collision.shape[0])
            is_collision = np.isin(all_indices, non_collision_idxs.flatten(), invert=True)

            collision_idxs = np.where(is_collision)[0]

            non_collision_sample_size = min(len(non_collision_idxs), n_sample_points//4*3) # sample non-collision points first
            collision_sample_size = min(len(collision_idxs), n_sample_points - non_collision_sample_size)

            non_collision_sample_idxs = np.random.choice(len(non_collision_idxs), size=non_collision_sample_size, replace=False)
            collision_sample_idxs = np.random.choice(len(collision_idxs), size=collision_sample_size, replace=False)

            # Combine the sampled indices
            idxs = np.concatenate((non_collision_idxs[non_collision_sample_idxs], collision_idxs[collision_sample_idxs]))
           
            # idxs = np.random.choice(len(points), min(max(int(len(points)/4),300),len(points)), replace=False)


            grasp_points_list.append(points[idxs])
            grasp_offsets_list.append(offsets[idxs])
            collision = collision[idxs].copy()
            scores = scores[idxs].copy()
            # scores[collision] = 0
            grasp_scores_list.append(scores)
            tolerance = tolerance[idxs].copy()
            # tolerance[collision] = 0
            grasp_tolerance_list.append(tolerance)
            grasp_collision_list.append(collision)


        if self.augment:
            cloud_sampled, object_poses_list = self.augment_data(cloud_sampled, object_poses_list)

        # delete unused memory
        del cloud, cloud_masked, color_masked, seg_masked, seg, depth, depth_mask, workspace_mask
        ret_dict = {}
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['cloud_colors'] = color_sampled.astype(np.float32)
        ret_dict['objectness_label'] = objectness_label.astype(np.int64)
        ret_dict['object_poses_list'] = object_poses_list
        ret_dict['grasp_points_list'] = grasp_points_list
        ret_dict['grasp_offsets_list'] = grasp_offsets_list
        ret_dict['grasp_labels_list'] = grasp_scores_list
        ret_dict['grasp_tolerance_list'] = grasp_tolerance_list
        ret_dict['grasp_collisions_list'] = grasp_collision_list
        return ret_dict

def load_grasp_labels_gc6d(root):
    obj_names = list(range(69, 201))
    valid_obj_idxs = []
    grasp_labels = {}
    for i, obj_name in enumerate(tqdm(obj_names, desc='Loading grasping labels...')):
        if obj_name not in list(range(69, 80)):
            continue
        valid_obj_idxs.append(obj_name) #here align with label png
        label = np.load(os.path.join(root, 'grasp_label', 'obj_{}_labels.npz'.format(str(obj_name).zfill(6))))
        tolerance = np.load(os.path.join(root, 'tolerance', 'obj_{}_tolerance.npy'.format(str(obj_name).zfill(6))))
        grasp_labels[obj_name] = (label['points'].astype(np.float32), label['offsets'].astype(np.float32),
                                label['scores'].astype(np.float32), tolerance)

    return valid_obj_idxs, grasp_labels

def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key:collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        return [[torch.from_numpy(sample) for sample in b] for b in batch]
    
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))

