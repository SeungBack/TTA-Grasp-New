import h5py
import open3d as o3d
import glob
import shutil
import os
from tqdm import tqdm
import numpy as np
import random

# get G1B_PATH from env import G1B_PATH
G1B_PATH = '/data/Grasp/GraspNet-1Billion'


splits = ['train', 'test_seen', 'test_similar', 'test_novel']
splits = ['test_seen']

for split in splits:
    input_path = os.path.join(G1B_PATH, 'grasp_qnet_final', split)
    
    # # Get all h5 files
    h5_paths = sorted(glob.glob(os.path.join(input_path, '*.h5')))
    print(f'Total number of h5 files in {split}:', len(h5_paths))
    
    if len(h5_paths) == 0:
        print(f"Warning: No .h5 files found in {input_path}")
        continue
    
    max_samples = 100
    
    # Group files by (obj_id, fric)
    file_dict = {}
    for h5_path in h5_paths:
        basename = os.path.basename(h5_path)
        parts = basename.split('_')
        
        # 파일명 형식 검증
        if len(parts) < 3:
            print(f"Warning: Unexpected filename format: {basename}")
            continue
            
        obj_id = parts[0]
        # parts[1]이 friction이라고 가정 (실제 형식에 맞게 수정 필요)
        fric = parts[1]  # 또는 parts[2], 실제 파일명 확인 필요
        
        key = (obj_id, fric)
        if key not in file_dict:
            file_dict[key] = []
        file_dict[key].append(h5_path)
    
    # Randomly select up to max_samples for each (obj_id, fric)
    selected_files = []
    files_to_remove = []
    
    for key, files in file_dict.items():
        if len(files) > max_samples:
            selected = random.sample(files, max_samples)
            removed = [f for f in files if f not in selected]
            files_to_remove.extend(removed)
        else:
            selected = files
        selected_files.extend(selected)
    
    print(f'Number of selected files after limiting to {max_samples} per (obj, fric): {len(selected_files)}')
    print(f'Number of files to remove: {len(files_to_remove)}')
    
    # # 실제로 파일 삭제 또는 이동 (주의: 백업 권장!)
    if len(files_to_remove) > 0:
        response = input(f"Delete {len(files_to_remove)} files? (yes/no): ")
        if response.lower() == 'yes':
            for f in tqdm(files_to_remove, desc="Removing files"):
                os.remove(f)
            print(f"Removed {len(files_to_remove)} files")
        else:
            print("Deletion cancelled")





# 
    # for h5_path in tqdm(h5_paths):

        # with h5py.File(h5_path, 'r') as f:
        #     obj_cloud = f['obj_cloud'][()]
        #     gripper_cloud = f['gripper_cloud'][()] 
        #     score = f['score'][()]
            
        # assert obj_cloud.shape[0] == 4096
        # assert gripper_cloud.shape[0] == 64
        
        # visualize
        # o3d_obj_cloud = o3d.geometry.PointCloud()
        # o3d_obj_cloud.points = o3d.utility.Vector3dVector(obj_cloud)
        # o3d_obj_cloud.paint_uniform_color([0, 1, 0])
        # o3d_gripper_cloud = o3d.geometry.PointCloud()
        # o3d_gripper_cloud.points = o3d.utility.Vector3dVector(gripper_cloud)
        # o3d_gripper_cloud.paint_uniform_color([1, 0, 0])
        # origin_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        # print('Score:', score)
        # o3d.visualization.draw_geometries([o3d_obj_cloud, origin_coord, o3d_gripper_cloud])
                                                              
