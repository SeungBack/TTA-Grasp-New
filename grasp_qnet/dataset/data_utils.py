# Codes from https://github.com/ldkong1205/PointCloud-C/blob/main/build/corrupt_utils.py

import numpy as np
import math


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


def downsample_point_cloud(cloud, num_points):
    # Original points
    original_points = cloud
    num_original_points = original_points.shape[0]

    # If the original number of points is already less than or equal to desired points
    if num_original_points <= num_points:
        print("Point cloud already has fewer points than required. Consider upsampling instead.")
        return cloud

    # Generate new points by random sampling
    indices = np.random.choice(num_original_points, num_points, replace=False)
    new_points = original_points[indices]

    return new_points

def to_fixed_size_pointcloud(pcd, num_points):
    """
    Convert a point cloud to a fixed size
    :param pcd: input point cloud
    :param num_points: number of points in the output point cloud
    :return: fixed size point cloud
    """
    if pcd.shape[0] > num_points:
        pcd = downsample_point_cloud(pcd, num_points)
    elif pcd.shape[0] < num_points:
        pcd = upsample_point_cloud(pcd, num_points)
    return pcd

def _shuffle_pointcloud(pcd):
    """
    Shuffle the points
    :param pcd: input point cloud
    :return: shuffled point clouds
    """
    idx = np.random.rand(pcd.shape[0], 1).argsort(axis=0)
    return np.take_along_axis(pcd, idx, axis=0)


def _gen_random_cluster_sizes(num_clusters, total_cluster_size):
    """
    Generate random cluster sizes
    :param num_clusters: number of clusters
    :param total_cluster_size: total size of all clusters
    :return: a list of each cluster size
    """
    rand_list = np.random.randint(num_clusters, size=total_cluster_size)
    cluster_size_list = [sum(rand_list == i) for i in range(num_clusters)]
    return cluster_size_list


def _sample_points(number_of_particles, min_xyz, max_xyz):
    # Add random noise to the point cloud
    x = np.random.uniform(min_xyz[0]*0.8, max_xyz[0]*1.2, (number_of_particles, 1))
    y = np.random.uniform(min_xyz[1]*0.8, max_xyz[1]*1.2, (number_of_particles, 1))
    z = np.random.uniform(min_xyz[2]*0.8, max_xyz[2]*1.2, (number_of_particles, 1))

    return np.concatenate([x, y, z], axis=1)

def corrupt_jitter(pointcloud, sigmas=[0.002, 0.00]):
    """
    Jitter the input point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    sigma = np.random.uniform(sigmas[0], sigmas[1], size=(pointcloud.shape[0], 3))
    pointcloud[:, :3] = pointcloud[:, :3] + sigma
    return pointcloud

def scale_and_translate(pointcloud, scale=[0.098, 1.02], translate=[-0.01, 0.01]):
    
    xyz1 = np.random.uniform(low=scale[0], high=scale[1], size=[3]) 
    xyz2 = np.random.uniform(low=translate[0], high=translate[1], size=[3])
    pointcloud[:, :3] = np.multiply(pointcloud[:, :3], xyz1) + xyz2
    return pointcloud

def rotate(pointcloud, angle=[-math.pi/6, math.pi/6]):
    """
    Rotate the input point cloud with x, y, z axes
    :param pointcloud: input point cloud
    :param angle: rotation angle in radians
    :return: rotated point cloud
    """
    angle_x = np.random.uniform(angle[0], angle[1])
    angle_y = np.random.uniform(angle[0], angle[1])
    angle_z = np.random.uniform(angle[0], angle[1])

    rotation_matrix_x = np.array([[1, 0, 0],
                                   [0, np.cos(angle_x), -np.sin(angle_x)],
                                   [0, np.sin(angle_x), np.cos(angle_x)]])

    rotation_matrix_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                                   [0, 1, 0],
                                   [-np.sin(angle_y), 0, np.cos(angle_y)]])

    rotation_matrix_z = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                                   [np.sin(angle_z), np.cos(angle_z), 0],
                                   [0, 0, 1]])

    rotation_matrix = rotation_matrix_z @ rotation_matrix_y @ rotation_matrix_x
    pointcloud[:, :3] = pointcloud[:, :3] @ rotation_matrix.T
    return pointcloud


def corrupt_dropout_global(pointcloud, drop_rates=[0.0, 0.25]):
    """
    Drop random points globally
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    drop_rate = np.random.uniform(drop_rates[0], drop_rates[1])
    num_points = pointcloud.shape[0]
    pointcloud = _shuffle_pointcloud(pointcloud)
    pointcloud = pointcloud[:int(num_points * (1 - drop_rate)), :]
    return pointcloud

from sklearn.neighbors import NearestNeighbors

def delete_random_knn_regions(point_cloud: np.ndarray, num_regions_to_delete: int, k_neighbors_for_region: int) -> tuple[np.ndarray, np.ndarray]:
    """
    주어진 포인트 클라우드에서 KNN으로 정의된 로컬 영역 중 일부를 랜덤하게 삭제합니다.

    각 포인트는 자신과 k_neighbors_for_region 개의 가장 가까운 이웃을 포함하는
    하나의 로컬 영역을 정의합니다. 이렇게 정의된 n개의 로컬 영역 중
    num_regions_to_delete 개를 무작위로 선택하여, 선택된 영역에 포함된
    모든 포인트들을 삭제합니다.

    Args:
        point_cloud (np.ndarray): 원본 포인트 클라우드. 형상은 [n, 3] 여야 합니다.
                                   n은 포인트의 수입니다.
        num_regions_to_delete (int): 삭제할 로컬 영역의 수 (m).
                                     1과 n 사이의 값이어야 합니다.
        k_neighbors_for_region (int): 각 로컬 영역을 정의하기 위해 사용할 이웃의 수 (k).
                                      0 이상이어야 합니다. k=0이면 영역은 포인트 자신만을 포함합니다.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - remaining_point_cloud (np.ndarray): 지정된 영역의 포인트들이 삭제된 후 남은 포인트 클라우드.
            - deleted_points_indices (np.ndarray): 삭제된 고유한 포인트들의 원본 인덱스 배열.

    Raises:
        TypeError: point_cloud가 NumPy 배열이 아닌 경우.
        ValueError: 입력 매개변수가 유효하지 않은 경우 (예: point_cloud 형상, m 또는 k 값).
    """
    if not isinstance(point_cloud, np.ndarray):
        raise TypeError("point_cloud는 NumPy 배열이어야 합니다.")
    if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
        raise ValueError("point_cloud는 [n, 3] 형상이어야 합니다.")

    num_points = point_cloud.shape[0]

    if not isinstance(num_regions_to_delete, int) or num_regions_to_delete <= 0:
        raise ValueError("삭제할 영역의 수(num_regions_to_delete)는 양의 정수여야 합니다.")
    if num_regions_to_delete > num_points:
        raise ValueError("삭제할 영역의 수(num_regions_to_delete)는 전체 포인트 수보다 클 수 없습니다.")
    if not isinstance(k_neighbors_for_region, int) or k_neighbors_for_region < 0:
        raise ValueError("영역 정의를 위한 이웃의 수(k_neighbors_for_region)는 0 이상의 정수여야 합니다.")

    if num_points == 0:
        return np.array([]).reshape(0, 3), np.array([])

    # 각 영역은 포인트 자신과 k_neighbors_for_region 개의 이웃을 포함합니다.
    # 따라서 NearestNeighbors에는 k_neighbors_for_region + 1 (자신 포함)을 전달합니다.
    # k_neighbors_for_region이 0이면, n_neighbors_for_knn_query는 1이 되어 자신만 찾습니다.
    n_neighbors_for_knn_query = k_neighbors_for_region + 1

    # n_neighbors_for_knn_query가 전체 포인트 수보다 클 수 없도록 합니다.
    # sklearn의 NearestNeighbors는 이를 자동으로 처리하지만, 명시적으로 하는 것이 좋습니다.
    actual_neighbors_to_find = min(n_neighbors_for_knn_query, num_points)

    # KNN 모델을 학습시키고 각 포인트의 이웃을 찾습니다.
    # neighbor_indices[i]는 i번째 포인트와 그 이웃들의 인덱스를 포함합니다.
    # 기본적으로 자기 자신이 첫 번째 이웃으로 포함됩니다.
    nbrs = NearestNeighbors(n_neighbors=actual_neighbors_to_find, algorithm='auto').fit(point_cloud)
    # distances, neighbor_indices = nbrs.kneighbors(point_cloud) # 거리 정보도 필요하면 사용
    neighbor_indices_list = nbrs.kneighbors(point_cloud, return_distance=False)

    # 삭제할 m개의 영역 중심(포인트 인덱스)을 무작위로 선택합니다.
    # np.random.choice는 중복 없이 선택합니다 (replace=False).
    region_center_indices_to_delete = np.random.choice(num_points, num_regions_to_delete, replace=False)

    # 선택된 m개의 영역에 속하는 모든 고유한 포인트 인덱스를 수집합니다.
    points_to_delete_set = set()
    for center_idx in region_center_indices_to_delete:
        # center_idx를 중심으로 하는 영역의 모든 포인트 인덱스
        # neighbor_indices_list[center_idx]는 center_idx 자신과 그 이웃들을 포함
        for point_idx in neighbor_indices_list[center_idx]:
            points_to_delete_set.add(point_idx)
    
    # 삭제할 포인트들의 인덱스 배열 (정렬은 선택 사항)
    final_points_to_delete_indices = np.array(list(points_to_delete_set))

    # 유지할 포인트에 대한 마스크를 생성합니다.
    mask_to_keep = np.ones(num_points, dtype=bool)
    if len(final_points_to_delete_indices) > 0: # 삭제할 인덱스가 있는 경우에만 마스킹
        mask_to_keep[final_points_to_delete_indices] = False

    remaining_point_cloud = point_cloud[mask_to_keep]
    
    return remaining_point_cloud



def corrupt_dropout_local(pointcloud, npoints=[0, 100]):
    """
    Randomly drop local clusters
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    num_points = pointcloud.shape[0]
    total_cluster_size = np.random.randint(npoints[0], npoints[1])
    num_clusters = np.random.randint(1, 8)
    cluster_size_list = _gen_random_cluster_sizes(num_clusters, total_cluster_size)
    for i in range(num_clusters):
        K = cluster_size_list[i]
        pointcloud = _shuffle_pointcloud(pointcloud)
        dist = np.sum((pointcloud - pointcloud[:1, :]) ** 2, axis=1, keepdims=True)
        idx = dist.argsort(axis=0)[::-1, :]
        pointcloud = np.take_along_axis(pointcloud, idx, axis=0)
        num_points -= K
        pointcloud = pointcloud[:num_points, :]
    return pointcloud


def corrupt_add_global(pointcloud, npoints=[0, 100]):
    """
    Add random points globally
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    npoints = np.random.randint(npoints[0], npoints[1])
    min_xyz = np.min(pointcloud, axis=0)
    max_xyz = np.max(pointcloud, axis=0)
    additional_pointcloud = _sample_points(npoints, min_xyz, max_xyz)
    pointcloud = np.concatenate([pointcloud, additional_pointcloud[:npoints]], axis=0)
    return pointcloud


def corrupt_add_local(pointcloud, npoints=[0, 100]):
    """
    Randomly add local clusters to a point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    num_points = pointcloud.shape[0]
    total_cluster_size = np.random.randint(npoints[0], npoints[1])
    num_clusters = np.random.randint(1, 8)
    cluster_size_list = _gen_random_cluster_sizes(num_clusters, total_cluster_size)
    pointcloud = _shuffle_pointcloud(pointcloud)
    add_pcd = np.zeros_like(pointcloud)
    num_added = 0
    for i in range(num_clusters):
        K = cluster_size_list[i]
        sigma = np.random.uniform(0.00075, 0.0025)
        add_pcd[num_added:num_added + K, :] = np.copy(pointcloud[i:i + 1, :])
        add_pcd[num_added:num_added + K, :] = add_pcd[num_added:num_added + K, :] + sigma * np.random.randn(
            *add_pcd[num_added:num_added + K, :].shape)
        num_added += K
    assert num_added == total_cluster_size
    dist = np.sum(add_pcd ** 2, axis=1, keepdims=True).repeat(3, axis=1)
    add_pcd[dist > 1] = add_pcd[dist > 1] / dist[dist > 1]  # ensure the added points are inside a unit sphere
    pointcloud = np.concatenate([pointcloud, add_pcd], axis=0)
    pointcloud = pointcloud[:num_points + total_cluster_size]
    return pointcloud