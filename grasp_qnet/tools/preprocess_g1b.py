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

obj_ids = { # 
    'train': [0, 2, 5, 7, 8, 9, 11, 14, 15, 17, 18, 20, 21, 22, 26, 27, 29, 30, 34, 36, 37, 38, 40, 41, 43, 44, 46, 48, 51, 52, 56, 57, 58, 60, 61, 62, 63, 66, 69, 70],
    'test_seen': [0, 2, 5, 7, 8, 9, 11, 14, 17, 18, 20, 21, 22, 26, 27, 29, 30, 38, 41, 48, 51, 52, 58, 60, 61, 62, 63, 66],
    'test_similar': [1, 3, 4, 6, 10, 12, 13, 16, 19, 23, 25, 35, 39, 42, 50, 53, 54, 59, 64, 65, 67, 68],
    'test_novel': [24, 28, 31, 32, 33, 45, 47, 49, 55, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87]
}
for key in obj_ids:
    obj_ids[key] = sorted(obj_ids[key])
    print('%s: %d objects'%(key, len(obj_ids[key])))



def transform_point_cloud(cloud, transform, format='4x4'):
    """ Transform points to new coordinates with transformation matrix.

        Input:
            cloud: [np.ndarray, (N,3), np.float32]
                points in original coordinates
            transform: [np.ndarray, (3,3)/(3,4)/(4,4), np.float32]
                transformation matrix, could be rotation only or rotation+translation
            format: [string, '3x3'/'3x4'/'4x4']
                the shape of transformation matrix
                '3x3' --> rotation matrix
                '3x4'/'4x4' --> rotation matrix + translation matrix

        Output:
            cloud_transformed: [np.ndarray, (N,3), np.float32]
                points in new coordinates
    """
    if not (format == '3x3' or format == '4x4' or format == '3x4'):
        raise ValueError('Unknown transformation format, only support \'3x3\' or \'4x4\' or \'3x4\'.')
    if format == '3x3':
        cloud_transformed = np.dot(transform, cloud.T).T
    elif format == '4x4' or format == '3x4':
        ones = np.ones(cloud.shape[0])[:, np.newaxis]
        cloud_ = np.concatenate([cloud, ones], axis=1)
        cloud_transformed = np.dot(transform, cloud_.T).T
        cloud_transformed = cloud_transformed[:, :3]
    return cloud_transformed


def upsample_point_cloud(cloud, num_points):
    # Original points
    original_points = cloud
    num_original_points = original_points.shape[0]

    # If the original number of points is already more than or equal to desired points
    if num_original_points >= num_points:
        print("Point cloud already has more points than required. Consider downsampling instead.")
        return cloud
    
    # Generate new points by random sampling
    indices = np.random.choice(num_original_points, num_points - num_original_points, replace=True)
    new_points = original_points[indices]

    # Combine original points and new points
    return np.vstack((original_points, new_points))

def generate_gripper_points(grasp):
    """Ultra-fast version with minimal operations"""
    width = grasp.width
    depth = grasp.depth
    
    # Pre-calculate constants
    tail_length = 0.04
    depth_base = 0.02
    half_width = width * 0.5
    
    # Pre-allocate with float32 for better cache performance
    points = np.zeros((64, 3), dtype=np.float32)
    
    # Pre-calculate step sizes (faster than np.linspace)
    finger_step = (depth + depth_base) / 19.0  # 20 points = 19 intervals
    connector_step = width / 13.0  # 14 points = 13 intervals
    tail_step = tail_length / 9.0  # 10 points = 9 intervals
    
    # Manual loop unrolling for fingers (most points)
    for i in range(20):
        x_val = -depth_base + i * finger_step
        points[i, 0] = x_val        # Left finger x
        points[i, 1] = -half_width  # Left finger y
        points[i+20, 0] = x_val     # Right finger x  
        points[i+20, 1] = half_width # Right finger y
    
    # Connector line
    for i in range(14):
        points[i+40, 0] = -depth_base
        points[i+40, 1] = -half_width + i * connector_step
    
    # Tail
    for i in range(10):
        points[i+54, 0] = -depth_base - i * tail_step
    
    return points

def process_with_retry(args, max_retries=100):
    for attempt in range(max_retries):
        try:
            return process(args)
        except Exception as e:
            print(f"Error processing {args} on attempt {attempt + 1}: {e}")
    return False

def process(args):
    out_dir, obj_id, target_fric, max_width, num_cloud_points, dataset_root = args

    plyfile = os.path.join(dataset_root, 'models', '%03d'%obj_id, 'nontextured.ply')
    model = o3d.io.read_point_cloud(plyfile)



    sampled_points, offsets, frics, collisions = get_model_grasps('%s/grasp_label/%03d_labels.npz'%(dataset_root, obj_id))
    frics[collisions] = -1
    frics[offsets[:, :, :, :, 2] > max_width] = -1  # filter by max width
    
    
    indices = np.argwhere(np.isclose(frics, target_fric, atol=1e-2))
    
    ind = indices[np.random.choice(indices.shape[0], 1, replace=False)][0]
    p, v, a, d = ind
    
    views = generate_views(300)
    view = views[v]
    angle, depth, width = offsets[p, v, a, d]
    grasp_rotation = viewpoint_params_to_matrix(-view, angle)
    grasp_point = sampled_points[p]
    score = 1.1 - target_fric
    
    grasp = np.array([score, width, 0.02, depth, *grasp_rotation.flatten(), *grasp_point])
    
    g = Grasp()
    g.grasp_array = grasp
    # o3d.visualization.draw_geometries([model, g.to_open3d_geometry()])
    
    obj_cloud_ = np.asarray(model.points)  # [n_points, 3]
    
    
    # transform scene to gripper frame
    target = obj_cloud_ - grasp_point[np.newaxis, :]  # [n_points, 3]
    target = np.matmul(target, grasp_rotation)  # [n_points, 3]

    ## crop the object in gripper closing area
    height = 0.06
    depth_base = 0.02
    depth_outer = 0.04
    width_offset = 0.02
    # gripper approach direction: 
    mask1 = ((target[:, 0] > -depth_base) & (target[:, 0] < depth + depth_outer))
    # width direction:
    mask2 = (target[:, 1] < (width/2 + width_offset)) & (target[:, 1] > -(width/2 + width_offset))
    # height direction:
    mask3 = ((target[:, 2] > -height/2) & (target[:, 2] < height/2))
    
    inner_mask = (mask1 & mask2 & mask3)
    obj_cloud_inner = obj_cloud_[inner_mask]
    
    # random sample n_points
    if obj_cloud_inner.shape[0] >= num_cloud_points:
        obj_cloud_inner = obj_cloud_inner[np.random.choice(obj_cloud_inner.shape[0], num_cloud_points, replace=False)]
    elif obj_cloud_inner.shape[0] < 512:
        return False
    else:
        obj_cloud_inner = upsample_point_cloud(obj_cloud_inner, num_cloud_points)
    
    gripper_points = generate_gripper_points(g)
    se3 = np.eye(4)
    se3[:3,:3] = grasp_rotation
    se3[:3,3] = grasp_point
    se3 = np.linalg.inv(se3)
    obj_cloud_inner = np.matmul(obj_cloud_inner, se3[:3,:3].T) + se3[:3,3]
    
    gripper_cloud = np.asarray(gripper_points)
    
    # o3d_cloud = o3d.geometry.PointCloud()
    # o3d_cloud.points = o3d.utility.Vector3dVector(obj_cloud_inner)
    # o3d_cloud.paint_uniform_color([0, 1, 0])
    # origin_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    # o3d_gripper = o3d.geometry.PointCloud()
    # o3d_gripper.points = o3d.utility.Vector3dVector(gripper_cloud)
    # o3d_gripper.paint_uniform_color([1, 0, 0])
    # o3d.visualization.draw_geometries([o3d_cloud, origin_coord, o3d_gripper])
    
    
    score = np.round(score, 2)
    
    save_path = os.path.join(out_dir, f'{obj_id}_{score:.1f}_{p}_{v}_{a}_{d}.h5')
    if os.path.exists(save_path):
        return False
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('obj_cloud', data=obj_cloud_inner)
            f.create_dataset('gripper_cloud', data=gripper_cloud)
            f.create_dataset('score', data=score)
            f.create_dataset('obj_id', data=obj_id)
        return True
    except Exception as e:
        print(f"Error saving file {save_path}: {e}")
        return False
    


    



dataset_root = '/home/seung/Datasets/GraspNet-1Billion'
dataset_root = '/data/Grasp/GraspNet-1Billion'
split = 'train'
num_samples_per_fric = 2000

split = 'test_seen'
num_samples_per_fric = 100

out_root = os.path.join(dataset_root, 'grasp_qnet_final', split)
max_attempt = 100
num_points = 4096
target_scores = np.linspace(0.1, 1.0, 10)

obj_ids = obj_ids[split]

# for obj_id in obj_ids:
#     print('Processing object %03d'%obj_id)
#     for target_fric in target_frics:
#         for _ in tqdm(range(num_samples_per_fric)):
#             for attempt in range(max_attempt):
#                 success = process(out_root, obj_id, target_fric=target_fric, max_width=0.14, num_cloud_points=num_points)
#                 if success:
#                     break

# 이미 존재하는 파일을 확인하고, 없는 작업만 수행하도록 수정
existing_files = glob.glob(os.path.join(out_root, '*.h5'))

# Parse existing files once into a lookup dictionary
existing_counts = defaultdict(int)
for f in existing_files:
    basename = os.path.basename(f)
    # Extract obj_id and target_fric from filename
    # 실제 형식: '{obj_id}_{score:.1f}_{p}_{v}_{a}_{d}.h5'
    parts = basename.split('_')
    if len(parts) >= 2:
        try:
            obj_id_str = parts[0]
            score_str = parts[1]
            
            # obj_id를 정수로 변환 (원래 obj_ids가 정수 리스트이므로)
            obj_id_int = int(obj_id_str)
            
            # score를 float으로 변환 후 문자열로 (동일한 형식 유지)
            score_float = float(score_str)
            
            key = (obj_id_int, f"{score_float:.1f}")
            existing_counts[key] += 1
        except (ValueError, IndexError):
            continue

print(f"Total existing files: {len(existing_files)}")
print(f"Unique (obj_id, score) combinations found: {len(existing_counts)}")
print(f"Sample existing_counts: {dict(list(existing_counts.items())[:10])}")

tasks = []
for obj_id in tqdm(obj_ids):
    for target_score in target_scores:
        # obj_id는 정수 그대로, target_score은 문자열로
        key = (obj_id, f"{target_score:.1f}")
        existing_count = existing_counts.get(key, 0)
        
        if existing_count >= num_samples_per_fric:
            print(f"Skipping obj_id {obj_id} with target_score {target_score:.1f}, already has {existing_count} files.")
            continue
        
        needed = num_samples_per_fric - existing_count
        print(f"obj_id {obj_id}, target_score {target_score:.1f}: existing {existing_count}, need {needed} more")
        target_fric = 1.1 - target_score  # Convert score back to friction
        for _ in range(needed):
            tasks.append((out_root, obj_id, target_fric, 0.14, num_points, dataset_root))

print(f"\nTotal tasks to process: {len(tasks)}")
        

# 멀티프로세싱 실행
with Pool(processes=5) as pool:
    results = list(tqdm(pool.imap(process_with_retry, tasks), total=len(tasks)))
    
print(f"Successfully generated {sum(results)} samples")