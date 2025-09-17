import numpy as np
import open3d as o3d
import random
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

def load_point_cloud(filename):
    """
    Load a point cloud from a file.
    """
    mesh = o3d.io.read_triangle_mesh(filename)
    pcd = mesh.sample_points_uniformly(50000)
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.01, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=20)
    return pcd

def visualize_point_cloud_with_normals(pcd, sample_ratio=0.05):
    """
    Visualize a point cloud with normals.
    """
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    
    # Sample a subset of points to visualize (for clarity)
    n_points = len(points)
    n_sample = int(n_points * sample_ratio)
    idx = random.sample(range(n_points), n_sample)
    
    sampled_points = points[idx]
    sampled_normals = normals[idx]
    
    # Create a mesh for visualization
    vis_pcd = o3d.geometry.PointCloud()
    vis_pcd.points = o3d.utility.Vector3dVector(sampled_points)
    vis_pcd.normals = o3d.utility.Vector3dVector(sampled_normals)
    
    # Visualize
    o3d.visualization.draw_geometries([vis_pcd], point_show_normal=True)

def is_antipodal(p1, n1, p2, n2, friction_cone_angle=np.pi/6):
    """
    Check if two points with their normals form an antipodal grasp.
    
    Args:
        p1, p2: 3D point coordinates
        n1, n2: Corresponding normals
        friction_cone_angle: Angle of the friction cone (default: 30 degrees)
    
    Returns:
        bool: True if points form an antipodal grasp
    """
    # Vector between the points
    grasp_direction = p2 - p1
    grasp_direction = grasp_direction / np.linalg.norm(grasp_direction)
    
    # Check if the normals are (approximately) opposite
    n1_dot_n2 = np.dot(n1, n2)
    if n1_dot_n2 > -0.8:  # Normals should be roughly opposite (cos(143°) = -0.8)
        return False
    
    # Check friction cone constraints
    cos_friction = np.cos(friction_cone_angle)
    
    # For p1, the grasp direction should be within the friction cone of -n1
    align1 = -np.dot(grasp_direction, n1)
    if align1 < cos_friction:
        return False
    
    # For p2, the opposite grasp direction should be within the friction cone of -n2
    align2 = np.dot(grasp_direction, n2)
    if align2 < cos_friction:
        return False
    
    return True

def filter_by_distance(grasps, min_distance, max_distance):
    """
    Filter grasps by distance between contact points.
    """
    filtered_grasps = []
    for grasp in grasps:
        p1, n1, p2, n2 = grasp
        distance = np.linalg.norm(p2 - p1)
        if min_distance <= distance <= max_distance:
            filtered_grasps.append(grasp)
    return filtered_grasps

def sample_antipodal_grasps(pcd, n_samples=1000, max_nn=100, min_distance=0.02, max_distance=0.08):
    """
    Sample antipodal grasps from a point cloud.
    
    Args:
        pcd: Open3D point cloud
        n_samples: Number of grasp candidates to sample
        max_nn: Maximum number of nearest neighbors to consider
        min_distance: Minimum distance between contact points
        max_distance: Maximum distance between contact points
    
    Returns:
        list: List of grasp candidates as (p1, n1, p2, n2) tuples
    """
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    
    # Build KD-tree for efficient neighbor search
    tree = KDTree(points)
    
    # Store valid antipodal grasps
    antipodal_grasps = []
    
    # Sample random points from the point cloud
    n_points = len(points)
    sampled_indices = random.sample(range(n_points), min(n_samples, n_points))
    
    for idx in sampled_indices:
        p1 = points[idx]
        n1 = normals[idx]
        
        # Get the general opposite direction based on the normal
        opposite_direction = -n1
        
        # Find points in the opposite direction
        # Query KD-tree for potential second contact points
        distances, indices = tree.query(p1, k=max_nn)
        
        for i, neighbor_idx in enumerate(indices):
            if i == 0:  # Skip the point itself
                continue
                
            p2 = points[neighbor_idx]
            n2 = normals[neighbor_idx]
            
            # Check if these points can form an antipodal grasp
            if is_antipodal(p1, n1, p2, n2):
                antipodal_grasps.append((p1, n1, p2, n2))
                break  # Found a valid grasp for this point, move to next
    
    # Filter grasps by distance
    antipodal_grasps = filter_by_distance(antipodal_grasps, min_distance, max_distance)
    
    return antipodal_grasps

def visualize_grasps(pcd, grasps, num_grasps_to_show=10):
    """
    Visualize antipodal grasps on the point cloud.
    """
    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add the point cloud
    vis.add_geometry(pcd)
    
    # Random sample from grasps if there are too many
    if len(grasps) > num_grasps_to_show:
        grasps = random.sample(grasps, num_grasps_to_show)
    
    # Add grasp visualizations as lines connecting antipodal points
    for grasp in grasps:
        p1, n1, p2, n2 = grasp
        
        # Create a line connecting the grasp points
        line = o3d.geometry.LineSet()
        points = o3d.utility.Vector3dVector([p1, p2])
        lines = o3d.utility.Vector2iVector([[0, 1]])
        line.points = points
        line.lines = lines
        line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Red lines
        
        # Add to visualization
        vis.add_geometry(line)
        
        # Visualize normals as well (optional)
        normal_length = 0.01
        
        # Normal at p1
        n1_line = o3d.geometry.LineSet()
        n1_line.points = o3d.utility.Vector3dVector([p1, p1 + normal_length * n1])
        n1_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        n1_line.colors = o3d.utility.Vector3dVector([[0, 1, 0]])  # Green for normals
        vis.add_geometry(n1_line)
        
        # Normal at p2
        n2_line = o3d.geometry.LineSet()
        n2_line.points = o3d.utility.Vector3dVector([p2, p2 + normal_length * n2])
        n2_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        n2_line.colors = o3d.utility.Vector3dVector([[0, 1, 0]])  # Green for normals
        vis.add_geometry(n2_line)
    
    # Run the visualization
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 1.0
    
    vis.run()
    vis.destroy_window()

def main():
    # Example usage
    # Load a point cloud (replace with your own path)
    pcd = load_point_cloud('/home/seung/Workspaces/Datasets/ACRONYM/meshes/1Shelves/1e3df0ab57e8ca8587f357007f9e75d1_rescaled.obj')
    
    # Visualize the point cloud with normals
    visualize_point_cloud_with_normals(pcd)
    
    # Sample antipodal grasps
    grasps = sample_antipodal_grasps(pcd, n_samples=1000, min_distance=0.02, max_distance=0.08)
    print(f"Found {len(grasps)} valid antipodal grasps")
    
    # Visualize grasps
    visualize_grasps(pcd, grasps, num_grasps_to_show=20)
    
    return grasps

if __name__ == "__main__":
    main()