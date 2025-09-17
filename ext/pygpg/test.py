
import pygpg
import numpy as np
import open3d as o3d
from graspnetAPI import GraspGroup
import time


# Example usage
if __name__ == "__main__":
    # Check for CUDA availability
     
    mesh_path = '/home/seung/Workspaces/Datasets/ACRONYM/meshes/1Shelves/1e3df0ab57e8ca8587f357007f9e75d1_rescaled.obj'
    
    
    # g_array is the format of numpy array is [score, width, height, depth, rotation_matrix(9), translation(3), object_id]
    # add slight noise in translation
    
    # load mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    obj_pcd = mesh.sample_points_uniformly(5000)
    points = np.asarray(obj_pcd.points)
    
    import time
    start_time = time.time()
    obj_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    fpfh_feat = o3d.pipelines.registration.compute_fpfh_feature(
                obj_pcd,
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=100)
            ).data
    print("Time taken to compute FPFH feature:", time.time() - start_time)
    print(fpfh_feat.shape)
            
    
    # points = np.loadtxt("example/box.txt")
    # obj_pcd = o3d.geometry.PointCloud()
    # obj_pcd.points = o3d.utility.Vector3dVector(points)
    
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
    gg= gg.nms(0.03, 30.0/180*np.pi)
    
    gg_o3d = gg.to_open3d_geometry_list()
    
    
    # o3d.visualization.draw_geometries([obj_pcd] + gg_o3d + geoms)
    o3d.visualization.draw_geometries([obj_pcd] + gg_o3d)