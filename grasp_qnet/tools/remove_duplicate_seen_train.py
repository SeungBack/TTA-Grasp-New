import os 
import numpy as np
import open3d as o3d
import h5py


from graspnetAPI import GraspNet
from graspnetAPI.utils.utils import generate_views, get_model_grasps, plot_gripper_pro_max, transform_points
from graspnetAPI.utils.rotation import viewpoint_params_to_matrix
from graspnetAPI.grasp import GraspGroup, Grasp
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import glob
from collections import defaultdict
import os




dataset_root = '/data/Grasp/GraspNet-1Billion'


train_root = os.path.join(dataset_root, 'grasp_qnet_final', 'train')
seen_root = os.path.join(dataset_root, 'grasp_qnet_final', 'test_seen')
num_points = 4096


# find the duplicate files b/w train and seen, and remove them from seen

train_h5_paths = sorted(glob.glob(train_root + '/*.h5'))
seen_h5_paths = sorted(glob.glob(seen_root + '/*.h5'))

print('Total number of train h5 files:', len(train_h5_paths))
print('Total number of seen h5 files:', len(seen_h5_paths))
train_basenames = set([os.path.basename(p) for p in train_h5_paths])
remove_count = 0
for seen_h5_path in tqdm(seen_h5_paths):
    seen_basename =  os.path.basename(seen_h5_path)
    if seen_basename in train_basenames:
        os.remove(seen_h5_path)
        remove_count += 1
    