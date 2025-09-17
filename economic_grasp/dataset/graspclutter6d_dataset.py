import os
import sys
import numpy as np
import scipy.io as scio
from PIL import Image
import pdb

import torch
import collections.abc as container_abcs
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image, \
    get_workspace_mask, remove_invisible_grasp_points
import json
import cv2
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

def get_workspace_mask_gc6d(cloud, seg, trans=None, organized=True, outlier=0):
    """ Keep points in workspace as input.

        Input:
            cloud: [np.ndarray, (H,W,3), np.float32]
                scene point cloud
            depth_mask: [np.ndarray, (H,W), np.bool]
                mask to indicate whether depth is valid
            seg: [np.ndarray, (H,W,), np.uint8]
                segmantation label of scene points
            trans: [np.ndarray, (4,4), np.float32]
                transformation matrix for scene points, default: None.
            organized: [bool]
                whether to keep the cloud in image shape (H,W,3)
            outlier: [float]
                if the distance between a point and workspace is greater than outlier, the point will be removed
                
        Output:
            workspace_mask: [np.ndarray, (H,W)/(H*W,), np.bool]
                mask to indicate whether scene points are in workspace
    """
    if organized:
        h, w, _ = cloud.shape
        cloud = cloud.reshape([h * w, 3])
        seg = seg.reshape(h * w)
        depth_mask = (cloud[:, 2] > 0) & (cloud[:, 2] < 1.2)

    if trans is not None:
        cloud = transform_point_cloud(cloud, trans)
    mask = (seg > 0) & depth_mask
    foreground = cloud[mask]
    
    xmin, ymin, zmin = foreground.min(axis=0)
    xmax, ymax, zmax = foreground.max(axis=0)
    mask_x = ((cloud[:, 0] > xmin - outlier) & (cloud[:, 0] < xmax + outlier))
    mask_y = ((cloud[:, 1] > ymin - outlier) & (cloud[:, 1] < ymax + outlier))
    mask_z = ((cloud[:, 2] > zmin - outlier) & (cloud[:, 2] < zmax + outlier))
    workspace_mask = (mask_x & mask_y & mask_z)
    if organized:
        workspace_mask = workspace_mask.reshape([h, w])

    return workspace_mask

class GraspClutter6DDataset(Dataset):
    def __init__(self, root, camera='kinect', split='train', num_points=20000,
                 voxel_size=0.005, remove_outlier=False, remove_invisible=True,
                 augment=False, load_label=True, return_raw_cloud=False):
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.remove_outlier = remove_outlier
        self.remove_invisible = remove_invisible
        self.camera = camera
        self.augment = augment
        self.load_label = load_label
        self.collision_labels = {}

        if split == 'all':
            with open(os.path.join(self.root, 'split_info', 'grasp_train_scene_ids.json')) as f:
                self.sceneIds = [int(x) for x in json.load(f)]
            with open(os.path.join(self.root, 'split_info', 'grasp_test_scene_ids.json')) as f:
                self.sceneIds += [int(x) for x in json.load(f)]
        elif split == 'train':
            with open(os.path.join(self.root, 'split_info', 'grasp_train_scene_ids.json')) as f:
                self.sceneIds = [int(x) for x in json.load(f)]
        elif split == 'test':
            with open(os.path.join(self.root, 'split_info', 'grasp_test_scene_ids.json')) as f:
                self.sceneIds = [int(x) for x in json.load(f)]
        else:
            raise ValueError('Invalid split: {} (must be train/test/all)'.format(split))
        self.sceneIds = ['{}'.format(str(x).zfill(6)) for x in self.sceneIds]#[:5] # shsh
        if self.camera == 'kinect':
            self.camera = 'azure-kinect' # for GraspNet-1Billon compatibility
        if self.camera == 'realsense':
            self.camera = 'realsense-d435' # for GraspNet-1Billon compatibility
        self.index_to_info = {}
        index = 0
        self.scene_ids = []
        for sceneid in tqdm(self.sceneIds, desc='Loading data path and collision labels...'):
            for ann_id in range(13):
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
        print('Total number of images for {}: {}'.format(split, len(self.scene_ids)))
        self.return_raw_cloud = return_raw_cloud

    def scene_list(self):
        return self.scene_ids

    def __len__(self):
        return len(self.index_to_info)

    def augment_data(self, point_clouds, object_poses_list):
        # Flipping along the YZ plane
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
            point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)

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
        seg_masked = seg[mask]

        if return_raw_cloud:
            return cloud_masked, color_masked
        
        ret_dict = {}
        if self.return_raw_cloud:
            ret_dict['point_clouds_raw'] = cloud_masked.astype(np.float32)
            ret_dict['cloud_colors_raw'] = color_masked.astype(np.float32)

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

        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['cloud_colors'] = color_sampled.astype(np.float32)
        ret_dict['coordinates_for_voxel'] = cloud_sampled.astype(np.float32) / self.voxel_size
        ret_dict['seg'] = seg_sampled.astype(np.float32)

        return ret_dict

    def get_data_label(self, index):
        #! TODO: this should be modified to load the grasp labels
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]

        graspness = np.load(self.graspnesspath[index])  # already remove outliers
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)

        # generate cloud
        # depth is in millimeters (mm), the transformed cloud is in meters (m).
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        depth_mask = (depth > 0)
        if self.remove_outlier:  # they are not the outliers, just the points far away from the objects
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
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
        graspness_sampled = graspness[idxs]
        objectness_label = seg_sampled.copy()
        segmentation_label = objectness_label.copy()
        objectness_label[objectness_label > 1] = 1

        object_poses_list = []
        grasp_points_list = []
        grasp_rotations_list = []
        grasp_depth_list = []
        grasp_widths_list = []
        grasp_scores_list = []
        view_graspness_list = []
        top_view_index_list = []

        # load labels
        grasp_labels = np.load(self.grasp_labels[scene])

        points = grasp_labels['points']
        rotations = grasp_labels['rotations'].astype(np.int32)
        depth = grasp_labels['depth'].astype(np.int32)
        scores = grasp_labels['scores'].astype(np.float32) / 10.
        widths = grasp_labels['widths'].astype(np.float32) / 1000.
        topview = grasp_labels['topview'].astype(np.int32)
        view_graspness = grasp_labels['vgraspness'].astype(np.float32)
        pointid = grasp_labels['pointid']
        for i, obj_idx in enumerate(obj_idxs):
            object_poses_list.append(poses[:, :, i])
            grasp_points_list.append(points[pointid == i])
            grasp_rotations_list.append(rotations[pointid == i])
            grasp_depth_list.append(depth[pointid == i])
            grasp_scores_list.append(scores[pointid == i])
            grasp_widths_list.append(widths[pointid == i])
            view_graspness_list.append(view_graspness[pointid == i])
            top_view_index_list.append(topview[pointid == i])

        if self.augment:
            cloud_sampled, object_poses_list = self.augment_data(cloud_sampled, object_poses_list)

        ret_dict = {}
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        # [scene_points, 3 (coords)]
        ret_dict['cloud_colors'] = color_sampled.astype(np.float32)
        # [scene_points, 3 (rgb)]
        ret_dict['coordinates_for_voxel'] = cloud_sampled.astype(np.float32) / self.voxel_size
        # [scene_points, 3 (coords)]
        ret_dict['graspness_label'] = graspness_sampled.astype(np.float32)
        # [scene_points, 1 (graspness)]
        ret_dict['objectness_label'] = objectness_label.astype(np.int64)
        # [scene_points, 1 (objectness)]
        ret_dict['segmentation_label'] = segmentation_label.astype(np.int64)
        # [scene_points, 1 (objectness)]
        ret_dict['object_poses_list'] = object_poses_list
        # list has a length of objects amount, each has size [3, 4] (pose matrix)
        ret_dict['grasp_points_list'] = grasp_points_list
        # list has a length of objects amount, each has size [object_points, 3 (coordinate)]
        ret_dict['grasp_rotations_list'] = grasp_rotations_list
        # list has a length of objects amount, each has size [object_points, 60 (view)]
        ret_dict['grasp_depth_list'] = grasp_depth_list
        # list has a length of objects amount, each has size [object_points, 60 (view)]
        ret_dict['grasp_widths_list'] = grasp_widths_list
        # list has a length of objects amount, each has size [object_points, 60 (view)]
        ret_dict['grasp_scores_list'] = grasp_scores_list
        # list has a length of objects amount, each has size [object_points, 60 (view)]
        ret_dict['view_graspness_list'] = view_graspness_list
        # list has a length of objects amount, each has size [object_points, 300 (view graspness)]
        ret_dict['top_view_index_list'] = top_view_index_list
        # list has a length of objects amount, each has size [object_points, top views index]

        return ret_dict


def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        return [[torch.from_numpy(sample) for sample in b] for b in batch]

    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))