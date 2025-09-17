import h5py
import open3d as o3d
import glob
import shutil
import os
from tqdm import tqdm
import numpy as np

# get G1B_PATH from env import G1B_PATH
input_path = '/home/seung/Workspaces/Datasets/ACRONYM/grasp_qnet'

# get all subdirectories in the input_path
h5_paths = sorted(glob.glob(input_path + '/*/*.h5'))

print('Total number of h5 files:', len(h5_paths))

for h5_path in tqdm(h5_paths):
    print(h5_path)

    with h5py.File(h5_path, 'r') as f:
        obj_cloud = f['obj_cloud'][()]
        gripper_cloud = f['gripper_cloud'][()] 
        score = f['score'][()]
        normal = f['normals'][()]
        
    assert obj_cloud.shape[0] == 1024
    assert gripper_cloud.shape[0] == 128
    assert normal.shape[0] == 1024