
import os
import sys
import numpy as np
import scipy.io as scio
from PIL import Image
import open3d as o3d
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.append('/home/seung/Workspaces/grasp/TestAdaGrasp/grasp_qnet')
from data_utils import CameraInfo, create_point_cloud_from_depth_image, transform_point_cloud, remove_invisible_grasp_points

from graspnetAPI.utils.utils import get_obj_pose_list, generate_views, get_model_grasps, transform_points
from graspnetAPI.utils.rotation import batch_viewpoint_params_to_matrix
from graspnetAPI import GraspNet, GraspGroup

from torch.utils.data import DataLoader,ConcatDataset, WeightedRandomSampler
import h5py

import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import glob


class ModelFreeCollisionDetector():
    """ Collision detection in scenes without object labels. Current finger width and length are fixed.

        Input:
                scene_points: [numpy.ndarray, (N,3), numpy.float32]
                    the scene points to detect
                voxel_size: [float]
                    used for downsample

        Example usage:
            mfcdetector = ModelFreeCollisionDetector(scene_points, voxel_size=0.005)
            collision_mask = mfcdetector.detect(grasp_group, approach_dist=0.03)
            collision_mask, iou_list = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05, return_ious=True)
            collision_mask, empty_mask = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05,
                                            return_empty_grasp=True, empty_thresh=0.01)
            collision_mask, empty_mask, iou_list = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05,
                                            return_empty_grasp=True, empty_thresh=0.01, return_ious=True)
    """
    def __init__(self, scene_points, voxel_size=0.005):
        self.finger_width = 0.01
        self.finger_length = 0.06
        self.voxel_size = voxel_size
        scene_cloud = o3d.geometry.PointCloud()
        scene_cloud.points = o3d.utility.Vector3dVector(scene_points)
        scene_cloud = scene_cloud.voxel_down_sample(voxel_size)
        self.scene_points = np.array(scene_cloud.points)

    def detect(self, grasp_group, approach_dist=0.03, collision_thresh=0.05, return_empty_grasp=False, empty_thresh=0.01, return_ious=False):
        """ Detect collision of grasps.

            Input:
                grasp_group: [GraspGroup, M grasps]
                    the grasps to check
                approach_dist: [float]
                    the distance for a gripper to move along approaching direction before grasping
                    this shifting space requires no point either
                collision_thresh: [float]
                    if global collision iou is greater than this threshold,
                    a collision is detected
                return_empty_grasp: [bool]
                    if True, return a mask to imply whether there are objects in a grasp
                empty_thresh: [float]
                    if inner space iou is smaller than this threshold,
                    a collision is detected
                    only set when [return_empty_grasp] is True
                return_ious: [bool]
                    if True, return global collision iou and part collision ious
                    
            Output:
                collision_mask: [numpy.ndarray, (M,), numpy.bool]
                    True implies collision
                [optional] empty_mask: [numpy.ndarray, (M,), numpy.bool]
                    True implies empty grasp
                    only returned when [return_empty_grasp] is True
                [optional] iou_list: list of [numpy.ndarray, (M,), numpy.float32]
                    global and part collision ious, containing
                    [global_iou, left_iou, right_iou, bottom_iou, shifting_iou]
                    only returned when [return_ious] is True
        """
        approach_dist = max(approach_dist, self.finger_width)
        T = grasp_group.translations
        R = grasp_group.rotation_matrices
        heights = grasp_group.heights[:,np.newaxis]
        depths = grasp_group.depths[:,np.newaxis]
        widths = grasp_group.widths[:,np.newaxis]
        targets = self.scene_points[np.newaxis,:,:] - T[:,np.newaxis,:]
        targets = np.matmul(targets, R)

        ## collision detection
        # height mask
        mask1 = ((targets[:,:,2] > -heights/2) & (targets[:,:,2] < heights/2))
        # left finger mask
        mask2 = ((targets[:,:,0] > depths - self.finger_length) & (targets[:,:,0] < depths))
        mask3 = (targets[:,:,1] > -(widths/2 + self.finger_width))
        mask4 = (targets[:,:,1] < -widths/2)
        # right finger mask
        mask5 = (targets[:,:,1] < (widths/2 + self.finger_width))
        mask6 = (targets[:,:,1] > widths/2)
        # bottom mask
        mask7 = ((targets[:,:,0] <= depths - self.finger_length)\
                & (targets[:,:,0] > depths - self.finger_length - self.finger_width))
        # shifting mask
        mask8 = ((targets[:,:,0] <= depths - self.finger_length - self.finger_width)\
                & (targets[:,:,0] > depths - self.finger_length - self.finger_width - approach_dist))

        # get collision mask of each point
        left_mask = (mask1 & mask2 & mask3 & mask4)
        right_mask = (mask1 & mask2 & mask5 & mask6)
        bottom_mask = (mask1 & mask3 & mask5 & mask7)
        shifting_mask = (mask1 & mask3 & mask5 & mask8)
        global_mask = (left_mask | right_mask | bottom_mask | shifting_mask)

        # calculate equivalant volume of each part
        left_right_volume = (heights * self.finger_length * self.finger_width / (self.voxel_size**3)).reshape(-1)
        bottom_volume = (heights * (widths+2*self.finger_width) * self.finger_width / (self.voxel_size**3)).reshape(-1)
        shifting_volume = (heights * (widths+2*self.finger_width) * approach_dist / (self.voxel_size**3)).reshape(-1)
        volume = left_right_volume*2 + bottom_volume + shifting_volume

        # get collision iou of each part
        global_iou = global_mask.sum(axis=1) / (volume+1e-6)

        # get collison mask
        collision_mask = (global_iou > collision_thresh)

        if not (return_empty_grasp or return_ious):
            return collision_mask

        ret_value = [collision_mask,]
        if return_empty_grasp:
            inner_mask = (mask1 & mask2 & (~mask4) & (~mask6))
            inner_volume = (heights * self.finger_length * widths / (self.voxel_size**3)).reshape(-1)
            empty_mask = (inner_mask.sum(axis=-1)/inner_volume < empty_thresh)
            ret_value.append(empty_mask)
        if return_ious:
            left_iou = left_mask.sum(axis=1) / (left_right_volume+1e-6)
            right_iou = right_mask.sum(axis=1) / (left_right_volume+1e-6)
            bottom_iou = bottom_mask.sum(axis=1) / (bottom_volume+1e-6)
            shifting_iou = shifting_mask.sum(axis=1) / (shifting_volume+1e-6)
            ret_value.append([global_iou, left_iou, right_iou, bottom_iou, shifting_iou])
        return ret_value

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

def load_grasp_labels(root, target_obj_ids=None):
    if target_obj_ids is not None:
        obj_names = target_obj_ids
    else:
        obj_names = list(range(88))
    print('Loading {} objects labels...'.format(len(obj_names)))
    valid_obj_idxs = []
    grasp_labels = {}
    for obj_name in tqdm(obj_names, desc='Loading grasping labels...'):
        if obj_name == 18: continue
        valid_obj_idxs.append(obj_name + 1) #here align with label png
        label = np.load(os.path.join(root, 'grasp_label', '{}_labels.npz'.format(str(obj_name).zfill(3))))
        grasp_labels[obj_name + 1] = (label['points'].astype(np.float32),
                               label['offsets'].astype(np.float32),
                                label['scores'].astype(np.float32))
        
    return valid_obj_idxs, grasp_labels




def process_grasp_data(args):
    
    global graspnet, grasp_labels, valid_obj_idxs
    dataset_root = '/home/seung/Datasets/GraspNet-1Billion'

    """Process a single grasp data item"""
    # (scene_name, img_num, obj_id, obj_idx, dataset_root, camera, out_dir, 
    #  valid_obj_idxs, grasp_labels, num_cloud_points, num_gripper_points) = item_data
    num_cloud_points = 1024
    split, camera, out_dir = args

    if split == 'train':
        scene_ids = list(range(100))
    elif split == 'test':
        scene_ids = list(range(100, 190))
    elif split == 'test_seen':
        scene_ids = list(range(100, 130))
    elif split == 'test_similar':
        scene_ids = list(range(130, 160))
    elif split == 'test_novel':
        scene_ids = list(range(160, 190))
    else:
        raise ValueError(f"Invalid split: {split}")
    
    # select random scene_id
    scene_id = np.random.choice(scene_ids, 1)[0]
    scene_name = f'scene_{str(scene_id).zfill(4)}'
    with open(os.path.join(dataset_root, 'scenes', scene_name, 'object_id_list.txt')) as f:
        obj_ids = [int(line.strip()) for line in f.readlines()]
    valid_obj_ids = []
    obj_idxs = []
    for obj_idx, obj_id in enumerate(obj_ids):
        if obj_id + 1 in valid_obj_idxs:
            valid_obj_ids.append(obj_id + 1)
            obj_idxs.append(obj_idx)
    
    # select random object
    rand_idx = np.random.choice(len(valid_obj_ids), 1)[0]
    obj_id = valid_obj_ids[rand_idx]
    obj_idx = obj_idxs[rand_idx]
    img_num = np.random.choice(256, 1)[0]
    

    depth_path = os.path.join(dataset_root, 'scenes', scene_name, camera, 'depth', str(img_num).zfill(4)+'.png')
    label_path = os.path.join(dataset_root, 'scenes', scene_name, camera, 'label', str(img_num).zfill(4)+'.png')
    meta_path = os.path.join(dataset_root, 'scenes', scene_name, camera, 'meta', str(img_num).zfill(4)+'.mat')
    depth = np.array(Image.open(depth_path))
    seg = np.array(Image.open(label_path))
    meta = scio.loadmat(meta_path)
    poses = meta['poses']
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

    # generate scene cloud
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # get valid points
    depth_mask = (depth > 0)
    mask = depth_mask
    cloud_masked = cloud[mask]
    seg_masked = seg[mask]
    
    
    if obj_id not in valid_obj_idxs:
        return None
    if (seg_masked == obj_id).sum() < 50:
        return None

    # load grasp labels
    pose = poses[:, :, obj_idx]
    points, offsets, fric_coefs = grasp_labels[obj_id]
    collision_labels = np.load(os.path.join(dataset_root, 'collision_label', scene_name,  'collision_labels.npz'))
    collision = collision_labels['arr_{}'.format(obj_idx)] #(Np, V, A, D)
    obj_cloud = cloud_masked[seg_masked==obj_id]

    # remove invisible grasp points
    visible_mask = remove_invisible_grasp_points(obj_cloud, points, poses[:,:,obj_idx], th=0.05)
    points = points[visible_mask]
    offsets = offsets[visible_mask]
    fric_coefs = fric_coefs[visible_mask]
    collision = collision[visible_mask]
    fric_coefs[collision] = 0

    # load grasp for the target object
    num_views, num_angles, num_depths = 300, 12, 4
    template_views = generate_views(num_views)
    template_views = template_views[np.newaxis, :, np.newaxis, np.newaxis, :]
    template_views = np.tile(template_views, [1, 1, num_angles, num_depths, 1])

    point_inds = np.arange(points.shape[0])
    num_points = len(point_inds)
    target_points = points[:, np.newaxis, np.newaxis, np.newaxis, :]
    target_points = np.tile(target_points, [1, num_views, num_angles, num_depths, 1])
    views = np.tile(template_views, [num_points, 1, 1, 1, 1]) # {N, 300, 12, 4, 3}
    angles = offsets[:, :, :, :, 0]
    depths = offsets[:, :, :, :, 1]
    widths = offsets[:, :, :, :, 2]

    target_points = target_points.reshape((-1,3))
    views = views.reshape((-1,3))
    angles = angles.reshape((-1))
    depths = depths.reshape((-1))
    widths = widths.reshape((-1))
    fric_coefs = fric_coefs.reshape((-1))
    scores = (1.1 - fric_coefs).reshape(-1,1)
    scores[fric_coefs == 0] = 0
    Rs = batch_viewpoint_params_to_matrix(-views, angles)

    num_grasp = widths.shape[0]
    widths = widths.reshape(-1,1)
    GRASP_HEIGHT = 0.02
    heights = GRASP_HEIGHT * np.ones((num_grasp,1))
    depths = depths.reshape(-1,1)
    rotations = Rs.reshape((-1,9))
    object_ids = obj_idx * np.ones((num_grasp,1), dtype=np.int32)

    # random select n_grasps
    # uniformly sample grasp points from score
    for k in range(10):
            
        target_score = np.random.choice(np.arange(0.1, 1.1, 0.1), 1)
        target_idxs = np.where(np.isclose(scores, target_score, atol=0.001))[0]
        target_idxs = target_idxs[widths[target_idxs].reshape(-1) <= 0.1 + 1e-5]
        if len(target_idxs) == 0:
            print('No valid grasp found for score {}, try again'.format(target_score))
            continue
        target_idx = np.random.choice(target_idxs, 1)
        target_point = target_points[target_idx]
        score = scores[target_idx]

        width = widths[target_idx]
        height = heights[target_idx]
        depth = depths[target_idx]
        rotation = rotations[target_idx]
        object_id = object_ids[target_idx]
        grasp = np.hstack([score, width, height, depth, rotation, target_point, object_id]).astype(np.float32) # [1, 17]

        # transform scene to gripper frame
        inv_pose = np.eye(4)
        inv_pose[:3, :4] = pose
        inv_pose = np.linalg.inv(inv_pose)[:3,:4]
        obj_cloud_ = transform_point_cloud(obj_cloud, inv_pose, '3x4')

        ## parse grasp parameters
        grasp_points = grasp[:, 13:16]
        grasp_poses = grasp[:, 4:13].reshape([-1,3,3])
        grasp_depths = grasp[:, 3]
        grasp_widths = grasp[:, 1]

        # transform scene to gripper frame
        target = (obj_cloud_[np.newaxis,:,:] - grasp_points[:,np.newaxis,:])
        target = np.matmul(target, grasp_poses)

        ## crop the object in gripper closing area
        height = 0.06
        depth_base = 0.02
        depth_outer = 0.05
        mask1 = ((target[:,:,2]>-height/2) & (target[:,:,2]<height/2))
        mask2 = ((target[:,:,0]>-depth_base) & (target[:,:,0]<grasp_depths[:,np.newaxis] + depth_outer))
        mask4 = (target[:,:,1]<-grasp_widths[:,np.newaxis]/2)
        mask6 = (target[:,:,1]>grasp_widths[:,np.newaxis]/2)
        inner_mask = (mask1 & mask2 &(~mask4) & (~mask6)) # [n_batch, n_points]
        obj_cloud_inner = obj_cloud_[inner_mask[0]]

        # random sample n_points
        if obj_cloud_inner.shape[0] >= num_cloud_points:
            obj_cloud_inner = obj_cloud_inner[np.random.choice(obj_cloud_inner.shape[0], num_cloud_points, replace=False)]
        elif obj_cloud_inner.shape[0] < 512:
            continue
        else:
            obj_cloud_inner = upsample_point_cloud(obj_cloud_inner, num_cloud_points)
        
        # sample gripper points
        gg = GraspGroup(grasp)
        mfcdetector = ModelFreeCollisionDetector(obj_cloud_inner, voxel_size=0.01)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=0.01)
        g = gg[0]
        if collision_mask.sum() == 1:
            continue
            # g.score = 0
            # print('Collision detected')
            
        gripper_points = generate_gripper_points(g)
        gripper_cloud = np.asarray(gripper_points)
        se3 = np.eye(4)
        se3[:3,:3] = g.rotation_matrix.reshape(3,3)
        se3[:3,3] = g.translation
        se3 = np.linalg.inv(se3)
        obj_cloud_inner = np.matmul(obj_cloud_inner, se3[:3,:3].T) + se3[:3,3]
        
        # gripper_pcd_new = o3d.geometry.PointCloud()
        # gripper_pcd_new.points = o3d.utility.Vector3dVector(gripper_points)
        # gripper_pcd_new.paint_uniform_color([0, 0, 1])  # red
        # obj_cloud_inner_pcd = o3d.geometry.PointCloud()
        # obj_cloud_inner_pcd.points = o3d.utility.Vector3dVector(obj_cloud_inner)
        # obj_cloud_inner_pcd.paint_uniform_color([1, 0, 0])  # green
        # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([obj_cloud_inner_pcd, coord_frame, gripper_pcd_new])

        obj_cloud_inner = obj_cloud_inner.astype(np.float32)
        gripper_cloud = gripper_cloud.astype(np.float32)
        score_val = np.round(g.score, 2).astype(np.float32)

        # Create unique filename
        target_fric_coef_val = float(target_score[0])
        target_idx_val = int(target_idx[0])
        save_path = os.path.join(out_dir, f'{scene_name}_{img_num:06d}_{obj_id:04d}_{target_fric_coef_val:.1f}_{target_idx_val:08d}.h5')
        
        # Create atomic file write through temporary file
        temp_path = f"{save_path}.tmp"
        with h5py.File(temp_path, 'w') as f:
            f.create_dataset('obj_cloud', data=obj_cloud_inner)
            f.create_dataset('gripper_cloud', data=gripper_cloud)
            f.create_dataset('score', data=score_val)
            f.create_dataset('scene_id', data=np.string_(scene_name))
            f.create_dataset('img_id', data=img_num)
            f.create_dataset('obj_id', data=obj_id)
        
        # Atomic rename to avoid partial files
        os.rename(temp_path, save_path)
        
    return 1


# def init_worker():
#     """Initialize worker process"""
#     # Disable Open3D logging in worker processes
#     o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    
#     # Set lower thread count for NumPy in each worker to prevent thread explosion
#     import os
#     os.environ["OMP_NUM_THREADS"] = "1"
#     os.environ["MKL_NUM_THREADS"] = "1"
#     os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
#     # NumPy doesn't have set_num_threads directly
#     try:
#         import mkl
#         mkl.set_num_threads(1)
#     except ImportError:
#         pass

# Global variables for worker processes
graspnet = None
grasp_labels = None
valid_obj_idxs = None

def init_worker(dataset_root, camera, split):
    """Initialize each worker process with shared data"""
    global graspnet, grasp_labels, valid_obj_idxs
    
    # Suppress output in worker processes
    
    print("Initializing worker...")
    
    # Initialize GraspNet for this worker
    graspnet = GraspNet(dataset_root, camera=camera, split=split)
    
    # Load grasp labels once per worker
    valid_obj_idxs, grasp_labels = load_grasp_labels(dataset_root, graspnet.getObjIds(graspnet.getSceneIds()))
    
    print(f"Worker initialized with {len(grasp_labels)} grasp labels")


def main():
    # Configuration
    dataset_root = '/home/seung/Datasets/GraspNet-1Billion'
    
    out_root = dataset_root + '/grasp_qnet_new'
    camera = 'realsense'
    split = 'train'  # Options: train, test, test_seen, test_similar, test_novel
    max_samples = 200000  # Maximum number of samples to generate
    num_processes = 4  # 더 많은 프로세스 사용
    
    out_dir = os.path.join(out_root, camera, split)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")
    
    # Get existing file count
    existing_count = len([f for f in os.listdir(out_dir) if f.endswith('.h5')])
    if existing_count > 0:
        print(f"Found {existing_count} existing samples. Will generate up to {max_samples - existing_count} more.")
    
    # Get scene list for task distribution
    print("Getting scene list...")
    temp_g = GraspNet(dataset_root, camera=camera, split=split)
    scene_ids = temp_g.getSceneIds()
    print(f"Found {len(scene_ids)} scenes")
    
    # Set up multiprocessing
    print(f"Starting {num_processes} worker processes")
    
    # Create process pool with initializer
    with mp.Pool(
        processes=num_processes, 
        initializer=init_worker,
        initargs=(dataset_root, camera, split)
    ) as pool:
        
        # Create tasks - one per scene
        scene_tasks = [(split, camera, out_dir) for _ in range(max_samples)]  # 원
        
        try:
            import time
            start_time = time.time()
            
            # Process tasks in parallel
            print(f"Processing {len(scene_tasks)} tasks...")
            print("Starting sample generation...")
            
            results = []
            processed_count = 0
            
            # 배치 단위로 처리하여 진행률 확인
            batch_size = min(100, len(scene_tasks))
            
            for i in range(0, len(scene_tasks), batch_size):
                batch_tasks = scene_tasks[i:i+batch_size]
                batch_results = pool.map(process_grasp_data, batch_tasks)
                
                # 결과 수집
                successful_results = [r for r in batch_results if r is not None]
                results.extend(successful_results)
                processed_count += len(batch_tasks)
                
                # 진행률 및 예상 시간 계산
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                if processed_count > 0:
                    avg_time_per_task = elapsed_time / processed_count
                    remaining_tasks = len(scene_tasks) - processed_count
                    estimated_remaining_time = avg_time_per_task * remaining_tasks
                    
                    # 시간 포맷팅
                    def format_time(seconds):
                        hours = int(seconds // 3600)
                        minutes = int((seconds % 3600) // 60)
                        secs = int(seconds % 60)
                        if hours > 0:
                            return f"{hours}h {minutes}m {secs}s"
                        elif minutes > 0:
                            return f"{minutes}m {secs}s"
                        else:
                            return f"{secs}s"
                    
                    success_rate = len(successful_results) / len(batch_tasks) * 100
                    total_samples = len(results) + existing_count
                    
                    print(f"Progress: {processed_count}/{len(scene_tasks)} tasks "
                        f"({processed_count/len(scene_tasks)*100:.1f}%) | "
                        f"Generated: {len(results)} samples | "
                        f"Total: {total_samples} | "
                        f"Success rate: {success_rate:.1f}% | "
                        f"Elapsed: {format_time(elapsed_time)} | "
                        f"Estimated remaining: {format_time(estimated_remaining_time)}")
                
                # 목표 달성 시 중단
                if total_samples >= max_samples:
                    print(f"Reached target of {max_samples} samples. Stopping.")
                    pool.terminate()
                    break
            
            total_time = time.time() - start_time
            print(f"\nProcessing complete in {format_time(total_time)}!")
            
            
        except KeyboardInterrupt:
            print("Interrupted by user. Terminating workers...")
            pool.terminate()
            pool.join()

if __name__ == "__main__":
    # Import in main function to reduce overhead
    import os
    import multiprocessing as mp
    
    # Configure NumPy for better performance
    os.environ["OMP_NUM_THREADS"] = "1"  # Limit threads at the main level too
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    # NumPy doesn't have set_num_threads directly
    try:
        import mkl
        mkl.set_num_threads(1)
    except ImportError:
        pass
    
    # Set multiprocessing start method to 'spawn' for better stability
    # especially when using large data structures like image processing
    if hasattr(mp, 'set_start_method'):
        try:
            mp.set_start_method('spawn')  # More stable than 'fork'
        except RuntimeError:
            # Method already set
            pass
            
    try:
        main()
    except KeyboardInterrupt:
        print("Process interrupted by user")
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        import traceback
        traceback.print_exc()