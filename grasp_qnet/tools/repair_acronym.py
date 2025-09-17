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
    print('Processing:', h5_path)
    score_path = h5_path.replace('grasp_qnet', 'meshes')
    # get directory name
    dir_name = os.path.dirname(score_path)
    # get file name
    file_name = os.path.basename(score_path).split('_')[0]
    index = int(os.path.basename(score_path).split('_')[1].split('.')[0])
    score_path = os.path.join(dir_name, file_name + '_scores.npy')
    scores = np.load(score_path)
    score = scores[index]
    if score == -1:
        score = 0.0
    else:
        score = round(max(1.1 - score, 0.0), 2)
    
    exist = os.path.exists(h5_path)
    f = h5py.File(h5_path, 'r')
    obj_cloud = f['obj_cloud'][()] 
    gripper_cloud = f['gripper_cloud'][()] 
    f.close()
    
    f = h5py.File(h5_path, 'w')
    f.create_dataset('obj_cloud', data=obj_cloud)
    f.create_dataset('gripper_cloud', data=gripper_cloud)
    f.create_dataset('score', data=score)
    f.close()
    print('score:', score)
    
    assert obj_cloud.shape[0] == 1024, 'obj_cloud shape is not 1024'
    assert gripper_cloud.shape[0] == 128, 'gripper_cloud shape is not 128'