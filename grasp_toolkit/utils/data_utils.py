import numpy as np
import torch
import random

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def jitter_point_cloud(cloud, sigmas=[0.001, 0.002]):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 tensor, original batch of point clouds
        Return:
          BxNx3 tensor, jittered batch of point clouds
    """
    sigma = np.random.uniform(sigmas[0], sigmas[1])
    cloud = cloud + sigma * torch.randn(cloud.shape).to(cloud)
    return cloud


def get_aug_matrix(type):
    if type == 'none':
        return np.eye(3)
    elif type == 'hflip':
        return np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    elif type == 'vflip':
        return np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    else:
        raise ValueError('Augmentation type not supported: %s' % type)

def augment_cloud(cloud, type='jitter'):
    """
    Augment point cloud with different methods.
        cloud = torch.tensor (B, N, 3)
    """
    if type == 'none':
        return cloud, None
    elif type == 'jitter':
        return jitter_point_cloud(cloud), None
    elif type == 'hflip': # flip along YZ plane
        flip_mat = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], device=cloud.device).float()
        # Handle batched point clouds
        return torch.matmul(cloud, flip_mat.T), flip_mat
    elif type == 'vflip': # flip along XZ plane
        flip_mat = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]], device=cloud.device).float()
        # Handle batched point clouds
        return torch.matmul(cloud, flip_mat.T), flip_mat
    elif type == 'rotate':
        B = cloud.shape[0]
        # Create batch of rotation matrices
        rot_matrices = []
        for i in range(B):
            # -30 to 30 degree rotation
            rot_angle = (np.random.random()*np.pi/3) - np.pi/6 # -30 ~ +30 degree
            c, s = np.cos(rot_angle), np.sin(rot_angle)
            rot_mat = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], device=cloud.device).float()
            rot_matrices.append(rot_mat)
        rot_tensors = torch.stack(rot_matrices)
        # Apply rotation to each point cloud in the batch
        rotated_clouds = []
        for i in range(B):
            rotated_clouds.append(torch.matmul(cloud[i], rot_tensors[i]))
        return torch.stack(rotated_clouds), rot_matrices
    else:
        raise ValueError('Augmentation type not supported: %s' % type)

def transform_point_cloud(cloud, mat):
    """
    Apply transformation matrix to point cloud.
        cloud = torch.tensor (N, 3)
        mat = torch.tensor (3, 3)
    """
    if mat is None:
        return cloud
    elif isinstance(cloud, torch.Tensor):
        return torch.matmul(mat, cloud.T).T
    elif isinstance(cloud, np.ndarray):
        # !TODO: check if this is correct
        mat = mat.cpu().numpy()
        # return np.dot(mat, cloud.T).T
        return np.matmul(cloud, mat)

def sample_point_cloud(point_cloud, target_points=1024):
    """
    Sample a point cloud to have exactly target_points.
    If the point cloud has more points than target_points, randomly downsample.
    If the point cloud has fewer points than target_points, randomly duplicate points.
    
    Args:
        point_cloud: Torch tensor of shape (N, 3) where:
            N is number of points
            3 is the dimension of each point (x, y, z)
        target_points: Target number of points (default: 1024)
    
    Returns:
        Torch tensor of shape (B, target_points, 3)
    """
    num_points, _ = point_cloud.shape
    
    if num_points >= target_points:
        # Downsample: randomly select target_points without replacement
        indices = torch.randperm(num_points, device=point_cloud.device)[:target_points]
        return point_cloud[indices]
    else:
        return torch.cat([point_cloud, point_cloud[
            torch.randint(0, num_points, (target_points - num_points,))]])
    return point_cloud[:target_points]  # If fewer points, just return as is

