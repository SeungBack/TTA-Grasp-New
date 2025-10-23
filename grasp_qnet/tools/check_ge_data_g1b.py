import h5py
import open3d as o3d
import glob
import shutil
import os
from tqdm import tqdm
import numpy as np
import random

# get G1B_PATH from env import G1B_PATH
G1B_PATH = os.environ['G1B_PATH']

splits = ['train', 'test_seen', 'test_similar', 'test_novel']

for split in splits:
    # camera = 'realsense'
    # split = 'train'
    input_path = os.path.join(G1B_PATH, 'grasp_qnet_final', split)
    print(input_path)
    # remove the over 80k
    h5_paths = sorted(glob.glob(input_path + '/*.h5'))

    print('Total number of h5 files:', len(h5_paths))
    
    # # randomly sample 10k files
    # random.shuffle(h5_paths)
    # remove_targets = h5_paths[10000:]  # Keep only the first 10k files
    # print('Number of files to remove:', len(remove_targets))
    # for h5_path in remove_targets:
    #     os.remove(h5_path)

    for h5_path in tqdm(h5_paths):

        with h5py.File(h5_path, 'r') as f:
            obj_cloud = f['obj_cloud'][()]
            gripper_cloud = f['gripper_cloud'][()] 
            score = f['score'][()]
            
        assert obj_cloud.shape[0] == 4096
        assert gripper_cloud.shape[0] == 64
        
        # visualize
        o3d_obj_cloud = o3d.geometry.PointCloud()
        o3d_obj_cloud.points = o3d.utility.Vector3dVector(obj_cloud)
        o3d_obj_cloud.paint_uniform_color([0, 1, 0])
        o3d_gripper_cloud = o3d.geometry.PointCloud()
        o3d_gripper_cloud.points = o3d.utility.Vector3dVector(gripper_cloud)
        o3d_gripper_cloud.paint_uniform_color([1, 0, 0])
        origin_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        print('Score:', score)
        o3d.visualization.draw_geometries([o3d_obj_cloud, origin_coord, o3d_gripper_cloud])
                                                              
    break
