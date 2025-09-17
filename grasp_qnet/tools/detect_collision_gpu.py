import numpy as np
import open3d as o3d
import torch
from graspnetAPI import GraspGroup


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




class GPUModelFreeCollisionDetector():


    def __init__(self, scene_points, voxel_size=0.005, device='cuda'):
        self.finger_width = 0.01
        self.finger_length = 0.06
        self.voxel_size = voxel_size
        self.device = torch.device(device) # Use torch.device object

        # Downsample point cloud using Open3D (CPU operation)
        scene_cloud = o3d.geometry.PointCloud()
        scene_cloud.points = o3d.utility.Vector3dVector(scene_points)
        scene_cloud = scene_cloud.voxel_down_sample(voxel_size)
        cpu_scene_points = np.array(scene_cloud.points)

        # Move scene points to GPU ONCE
        self.scene_points_tensor = torch.tensor(cpu_scene_points,
                                               dtype=torch.float32,
                                               device=self.device)
        self.num_points = self.scene_points_tensor.shape[0]
        print(f"Initialized GPU detector with {self.num_points} scene points on {self.device}")


    def detect(self, grasp_group, approach_dist=0.03, collision_thresh=0.05,
               return_empty_grasp=False, empty_thresh=0.01, return_ious=False):
        """ Detect collision of grasps using optimized GPU acceleration. """
        approach_dist = max(approach_dist, self.finger_width)

        # Get grasp parameters (NumPy arrays)
        T = grasp_group.translations
        R = grasp_group.rotation_matrices
        heights = grasp_group.heights
        depths = grasp_group.depths
        widths = grasp_group.widths
        num_grasps = T.shape[0]

        if num_grasps == 0:
             # Handle empty grasp group case
            ret_value = [np.array([], dtype=bool)]
            if return_empty_grasp:
                ret_value.append(np.array([], dtype=bool))
            if return_ious:
                ret_value.append([np.array([], dtype=np.float32)] * 5)
            return ret_value if len(ret_value) > 1 else ret_value[0]


        # === Transfer ALL grasp data to GPU ONCE ===
        T_tensor = torch.tensor(T, dtype=torch.float32, device=self.device)
        R_tensor = torch.tensor(R, dtype=torch.float32, device=self.device)
        # Unsqueeze later dimensions for broadcasting compatibility
        heights_tensor = torch.tensor(heights, dtype=torch.float32, device=self.device).unsqueeze(1)
        depths_tensor = torch.tensor(depths, dtype=torch.float32, device=self.device).unsqueeze(1)
        widths_tensor = torch.tensor(widths, dtype=torch.float32, device=self.device).unsqueeze(1)
        # ============================================

        # Perform computations entirely on GPU for ALL grasps
        # Note: This increases peak GPU memory usage. If you run out of memory,
        # you'll need to re-introduce batching, but structure it differently
        # (slice the GPU tensors, not transfer new data each time).

        # Transform points to grasp coordinate system
        # Shape: [num_grasps, num_points, 3]
        # scene_points_tensor shape: [num_points, 3] -> unsqueeze(0) -> [1, num_points, 3]
        # T_tensor shape: [num_grasps, 3] -> unsqueeze(1) -> [num_grasps, 1, 3]
        targets = self.scene_points_tensor.unsqueeze(0) - T_tensor.unsqueeze(1)

        # Batched matrix multiplication
        # targets shape: [num_grasps, num_points, 3]
        # R_tensor shape: [num_grasps, 3, 3]
        # Need R_tensor transpose for correct multiplication order if using matmul
        # transformed_targets = torch.matmul(targets, R_tensor.transpose(1, 2)) # Alternative
        transformed_targets = torch.einsum('bpi,bij->bpj', targets, R_tensor) # einsum is fine


        # Collision detection masks (compute in a single large batch)
        # Height mask
        mask1 = (transformed_targets[:,:,2] > -heights_tensor/2) & (transformed_targets[:,:,2] < heights_tensor/2)

        # Left finger region
        left_region = (transformed_targets[:,:,0] > depths_tensor - self.finger_length) & \
                      (transformed_targets[:,:,0] < depths_tensor) & \
                      (transformed_targets[:,:,1] > -(widths_tensor/2 + self.finger_width)) & \
                      (transformed_targets[:,:,1] < -widths_tensor/2)

        # Right finger region
        right_region = (transformed_targets[:,:,0] > depths_tensor - self.finger_length) & \
                       (transformed_targets[:,:,0] < depths_tensor) & \
                       (transformed_targets[:,:,1] < (widths_tensor/2 + self.finger_width)) & \
                       (transformed_targets[:,:,1] > widths_tensor/2)

        # Bottom region condition
        bottom_region_cond = (transformed_targets[:,:,0] <= depths_tensor - self.finger_length) & \
                         (transformed_targets[:,:,0] > depths_tensor - self.finger_length - self.finger_width)

        # Shifting region condition
        shifting_region_cond = (transformed_targets[:,:,0] <= depths_tensor - self.finger_length - self.finger_width) & \
                           (transformed_targets[:,:,0] > depths_tensor - self.finger_length - self.finger_width - approach_dist)

        # Shared width region for bottom and shifting
        width_region_both = (transformed_targets[:,:,1] > -(widths_tensor/2 + self.finger_width)) & \
                            (transformed_targets[:,:,1] < (widths_tensor/2 + self.finger_width))

        # Final Masks
        left_mask = mask1 & left_region
        right_mask = mask1 & right_region
        bottom_mask = mask1 & bottom_region_cond & width_region_both
        shifting_mask = mask1 & shifting_region_cond & width_region_both

        # Combine all masks to get global collision mask
        global_mask = left_mask | right_mask | bottom_mask | shifting_mask

        # Calculate equivalent volume of each part (on GPU)
        # Ensure shapes are compatible for broadcasting, squeeze unnecessary dims
        left_right_volume = (heights_tensor * self.finger_length * self.finger_width / (self.voxel_size**3)).squeeze(-1) # Shape [num_grasps]
        bottom_volume = (heights_tensor * (widths_tensor+2*self.finger_width) * self.finger_width / (self.voxel_size**3)).squeeze(-1) # Shape [num_grasps]
        shifting_volume = (heights_tensor * (widths_tensor+2*self.finger_width) * approach_dist / (self.voxel_size**3)).squeeze(-1) # Shape [num_grasps]
        # Ensure volumes are non-zero before division, add epsilon
        epsilon = 1e-6
        volume = left_right_volume * 2 + bottom_volume + shifting_volume + epsilon


        # Calculate IOUs (on GPU)
        # Sum over the points dimension (dim=1)
        global_iou = torch.sum(global_mask.float(), dim=1) / volume

        # Get collision mask (on GPU)
        collision_mask_tensor = (global_iou > collision_thresh)

        # --- Prepare results ---
        # Transfer final results back to CPU ONCE
        collision_mask = collision_mask_tensor.cpu().numpy()

        ret_value = [collision_mask,]

        if return_empty_grasp:
            # Inner region for empty grasp detection
            inner_region = (transformed_targets[:,:,0] > depths_tensor - self.finger_length) & \
                           (transformed_targets[:,:,0] < depths_tensor) & \
                           (transformed_targets[:,:,1] > -widths_tensor/2) & \
                           (transformed_targets[:,:,1] < widths_tensor/2)

            inner_mask = mask1 & inner_region
            # Ensure inner_volume has shape [num_grasps]
            inner_volume = (heights_tensor * self.finger_length * widths_tensor / (self.voxel_size**3)).squeeze(-1) + epsilon
            empty_mask_tensor = (torch.sum(inner_mask.float(), dim=1) / inner_volume < empty_thresh)
            ret_value.append(empty_mask_tensor.cpu().numpy())

        if return_ious:
            left_iou = torch.sum(left_mask.float(), dim=1) / (left_right_volume + epsilon)
            right_iou = torch.sum(right_mask.float(), dim=1) / (left_right_volume + epsilon)
            bottom_iou = torch.sum(bottom_mask.float(), dim=1) / (bottom_volume + epsilon)
            shifting_iou = torch.sum(shifting_mask.float(), dim=1) / (shifting_volume + epsilon)

            iou_list_cpu = [
                global_iou.cpu().numpy(),
                left_iou.cpu().numpy(),
                right_iou.cpu().numpy(),
                bottom_iou.cpu().numpy(),
                shifting_iou.cpu().numpy()
            ]
            ret_value.append(iou_list_cpu)

        # Return only the mask if no other options are selected, otherwise the list
        return ret_value[0] if len(ret_value) == 1 else ret_value

# Example usage
if __name__ == "__main__":
    # Check for CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
     
    gg_array_path = '/home/seung/Workspaces/Datasets/ACRONYM/meshes/1Shelves/1e3df0ab57e8ca8587f357007f9e75d1_gg.npy'
    mesh_path = '/home/seung/Workspaces/Datasets/ACRONYM/meshes/1Shelves/1e3df0ab57e8ca8587f357007f9e75d1_rescaled.obj'
    
    # load gg_array
    gg = GraspGroup()
    gg.grasp_group_array = np.load(gg_array_path)
    
    # g_array is the format of numpy array is [score, width, height, depth, rotation_matrix(9), translation(3), object_id]
    # add slight noise in translation
    gg.grasp_group_array[:, 12:15] += np.random.uniform(-0.02, 0.02, (gg.grasp_group_array.shape[0], 3))
    
    gg_o3d = gg.to_open3d_geometry_list()
    
    # load mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    obj_pcd = mesh.sample_points_uniformly(50000)
    obj_pcd = np.asarray(obj_pcd.points)
    
    # Compare CPU vs GPU performance
    import time
    
    # Original CPU version
    start_cpu = time.time()
    mfcdetector_cpu = ModelFreeCollisionDetector(obj_pcd, voxel_size=0.001)
    collision_mask_cpu = mfcdetector_cpu.detect(gg, approach_dist=0.05, collision_thresh=0.001)
    cpu_time = time.time() - start_cpu
    print(f"CPU Time taken: {cpu_time:.4f}s")
    print(f"CPU Collisions detected: {np.sum(collision_mask_cpu)}")
    
    # New GPU version
    start_gpu = time.time()
    mfcdetector_gpu = GPUModelFreeCollisionDetector(obj_pcd, voxel_size=0.001, device=device)
    collision_mask_gpu = mfcdetector_gpu.detect(gg, approach_dist=0.05, collision_thresh=0.001)
    gpu_time = time.time() - start_gpu
    print(f"GPU Time taken: {gpu_time:.4f}s")
    print(f"GPU Collisions detected: {np.sum(collision_mask_gpu)}")
    
    # Check if results match
    if isinstance(collision_mask_cpu, list):
        # If original returns a list (depending on params)
        matches = np.array_equal(collision_mask_cpu[0], collision_mask_gpu)
    else:
        matches = np.array_equal(collision_mask_cpu, collision_mask_gpu)
    
    print(f"Results match: {matches}")
    print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    
    # Visualization (same as original)
    obj_o3d = o3d.geometry.PointCloud()
    obj_o3d.points = o3d.utility.Vector3dVector(obj_pcd)
    obj_o3d.paint_uniform_color([0, 1, 0])  # green
    o3d.visualization.draw_geometries([obj_o3d] + gg_o3d)