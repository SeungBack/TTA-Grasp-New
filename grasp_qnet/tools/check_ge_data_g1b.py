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

cameras = ['kinect', 'realsense']
splits = ['test_seen', 'test_similar', 'test_novel']

for camera in cameras:
    for split in splits:
        # camera = 'realsense'
        # split = 'train'
        input_path = os.path.join(G1B_PATH, 'grasp_qnet', camera, split)
        # remove the over 80k
        h5_paths = sorted(glob.glob(input_path + '/*.h5'))

        print('Total number of h5 files:', len(h5_paths))
        
        # randomly sample 10k files
        random.shuffle(h5_paths)
        remove_targets = h5_paths[10000:]  # Keep only the first 10k files
        print('Number of files to remove:', len(remove_targets))
        for h5_path in remove_targets:
            os.remove(h5_path)

        # for h5_path in tqdm(h5_paths):

        #     with h5py.File(h5_path, 'r') as f:
        #         obj_cloud = f['obj_cloud'][()]
        #         gripper_cloud = f['gripper_cloud'][()] 
        #         score = f['score'][()]
                
        #     assert obj_cloud.shape[0] == 1024
        #     assert gripper_cloud.shape[0] == 128
