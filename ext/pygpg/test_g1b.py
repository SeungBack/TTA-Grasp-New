
import pygpg
import numpy as np
import open3d as o3d
from graspnetAPI import GraspGroup
import time
import scipy.io as scio
from PIL import Image

class CameraInfo():
    """ Camera intrisics for point cloud creation. """

    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale


def create_point_cloud_from_depth_image(depth, camera, organized=True):
    """ Generate point cloud using depth image only.

        Input:
            depth: [numpy.ndarray, (H,W), numpy.float32]
                depth image
            camera: [CameraInfo]
                camera intrinsics
            organized: bool
                whether to keep the cloud in image shape (H,W,3)

        Output:
            cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
    """
    assert (depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale  #
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud

# Example usage
if __name__ == "__main__":
    # Check for CUDA availability
     
    depth_path = '/home/seung/Workspaces/Datasets/GraspNet-1Billion/scenes/scene_0000/realsense/depth/0000.png'
    meta_path = '/home/seung/Workspaces/Datasets/GraspNet-1Billion/scenes/scene_0000/realsense/meta/0000.mat'
    
    depth = np.array(Image.open(depth_path))
    meta = scio.loadmat(meta_path)
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                    factor_depth)
    points = create_point_cloud_from_depth_image(depth, camera)
    points = points[depth > 0].reshape(-1, 3)
    points = np.asarray(points)
    points_o3d = o3d.geometry.PointCloud()
    points_o3d.points = o3d.utility.Vector3dVector(points)
    # points_o3d.farthest_point_down_sample(10000)
    points_o3d.voxel_down_sample(voxel_size=0.01)

    # remove plane points using RANSAC
    plane_model, inliers = points_o3d.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=100)
    points_o3d = points_o3d.select_by_index(inliers, invert=True)
    
    points = np.asarray(points_o3d.points)
    # random sample 10000 points
    points = points[np.random.choice(points.shape[0], 10000, replace=False)]

    
    num_grasp_samples = 100
    gripper_config_file = "gripper_params_g1b.cfg"
    start_time = time.time()
    grasps = pygpg.generate_grasps(points, num_grasp_samples, False, gripper_config_file)
    print("Time taken to generate grasps:", time.time() - start_time)
    gg_array = []
    geoms = []
    for grasp in grasps:
        bottom = grasp.get_grasp_bottom()
        top = grasp.get_grasp_top()
        surface = grasp.get_grasp_surface()
        approach = grasp.get_grasp_approach()
        binormal = grasp.get_grasp_binormal()
        axis = grasp.get_grasp_axis()
        width = grasp.get_grasp_width()
        
        # # visualize bottom, top, surface, approach, binormal, axis
        # bottom_pcd = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
        # bottom_pcd.paint_uniform_color([1, 0, 0])
        # bottom_pcd.translate(bottom)
        # top_pcd = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
        # top_pcd.paint_uniform_color([0, 1, 0])
        # top_pcd.translate(top)
        # surface_pcd = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
        # surface_pcd.paint_uniform_color([0, 0, 1])
        # surface_pcd.translate(surface)
        # geoms.append(bottom_pcd)
        # geoms.append(top_pcd)
        # geoms.append(surface_pcd)
        
        pose = np.eye(4)
        pose[:3, 0] = approach
        pose[:3, 1] = binormal
        pose[:3, 2] = axis
        contact = (top + bottom) / 2
        pose[:3, 3] = contact
        width = grasp.get_grasp_width()*2
        g_array = np.array([
            0.1, width, 0.02, 0.02, *pose[:3, :3].reshape(-1), *pose[:3, 3], -1
        ])
        gg_array.append(g_array)
    gg_array = np.array(gg_array)
    gg = GraspGroup(gg_array)
    
    gg_o3d = gg.to_open3d_geometry_list()
    
    
    o3d.visualization.draw_geometries([points_o3d] + gg_o3d)