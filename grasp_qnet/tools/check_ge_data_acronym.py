import h5py
import open3d as o3d
import glob
import os
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def process_h5_file(h5_path):
    """Process a single H5 file to add normals."""
    try:
        # Check if normals already exist
        with h5py.File(h5_path, 'r') as f:
            if 'normals' in f.keys():
                print(f'Normals already exist in {h5_path}')
                return False
            obj_cloud = f['obj_cloud'][()]
            gripper_cloud = f['gripper_cloud'][()] 
            score = f['score'][()]
        
        # Estimate normals
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_cloud)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30))
        try:
            pcd.orient_normals_consistent_tangent_plane(30)
        except Exception as e:
            print(f'Error in orienting normals for {h5_path}: {e}')
        normals = np.asarray(pcd.normals)
        
        # Save back to H5 file
        with h5py.File(h5_path, 'w') as f:
            f['obj_cloud'] = obj_cloud
            f['gripper_cloud'] = gripper_cloud
            f['score'] = score
            f['normals'] = normals
        
        return True
    except Exception as e:
        print(f"Error processing {h5_path}: {e}")
        return False

def main():
    # Get input path from environment variable or use the default
    input_path = '/home/seung/Workspaces/Datasets/ACRONYM/grasp_qnet'

    # Get all H5 files
    h5_paths = sorted(glob.glob(input_path + '/*/*.h5'))
    print(f'Total number of H5 files to process: {len(h5_paths)}')
    
    # Determine the number of processes to use
    num_processes = 8
    print(f'Using {num_processes} processes')
    
    # Create a pool of workers
    with mp.Pool(processes=num_processes) as pool:
        # Use tqdm to track progress
        results = list(tqdm(
            pool.imap(process_h5_file, h5_paths),
            total=len(h5_paths),
            desc="Processing H5 files"
        ))
    
    # Report statistics
    success_count = results.count(True)
    skipped_count = results.count(False)
    print(f"Processing complete: {success_count} files processed, {skipped_count} files skipped or failed")

if __name__ == "__main__":
    # Enable open3d multiprocessing support
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    mp.set_start_method('spawn', force=True)  # This is important for Open3D to work with multiprocessing
    main()