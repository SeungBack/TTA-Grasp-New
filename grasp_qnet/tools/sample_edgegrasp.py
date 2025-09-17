import open3d as o3d
import numpy as np
import open3d as o3d
import numpy as np
from graspnetAPI import Grasp, GraspGroup
import torch
from torch_geometric.nn import radius
import torch.nn.functional as F
import time

class FarthestSamplerTorch:
    def __init__(self):
        pass
    def _calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def __call__(self, pts, k):
        index_list = []
        farthest_pts = torch.zeros(k, 3).to(pts.device)
        index = np.random.randint(len(pts))
        farthest_pts[0] = pts[index]
        index_list.append(index)
        distances = self._calc_distances(farthest_pts[0], pts)
        for i in range(1, k):
            index = torch.argmax(distances)
            farthest_pts[i] = pts[index]
            index_list.append(index)
            distances = torch.minimum(distances, self._calc_distances(farthest_pts[i], pts))
        return farthest_pts, index_list


def orthognal_grasps_translate(geometry_mask,depth_projection,half_baseline,sample_normal,des_normals,sample_pos):

    '''
    :param geometry_mask: [bool,bool,,]
    :param depth_projection:
    :param sample_normal:
    :param des_normals:
    :param sample_pos:
    :return: mX4X4 matrices that used to execute grasp in simulation
    '''
    # if these is no reasonable points do nothing
    assert sum(geometry_mask)>0
    print('grasps_translate')
    depth = depth_projection[geometry_mask]
    # translate
    half_baseline = half_baseline[geometry_mask]

    translation = 0.023 - torch.abs(half_baseline)
    non_translation_mask = translation < 0.
    translation[non_translation_mask] = 0.
    translation = -translation

    # finger depth
    gripper_dis_from_source = (0.072-0.007 - depth).unsqueeze(dim=-1)
    z_axis = -sample_normal[geometry_mask]  # todo careful
    y_axis = des_normals[geometry_mask]
    x_axis = torch.cross(y_axis, z_axis,dim=1)
    x_axis = F.normalize(x_axis, p=2,dim=1)
    y_axis = torch.cross(z_axis, x_axis,dim=1)
    y_axis = F.normalize(y_axis, p=2,dim=1)
    gripper_position = gripper_dis_from_source.repeat(1, 3) * (-z_axis) + sample_pos[geometry_mask]
    transform_matrix = torch.cat((x_axis.unsqueeze(dim=-1), y_axis.unsqueeze(dim=-1),
                                  z_axis.unsqueeze(dim=-1), gripper_position.unsqueeze(dim=-1)), dim=-1)
    homo_agument = torch.as_tensor([0., 0., 0., 1.]).reshape(1, 1, 4).repeat(len(z_axis), 1, 1)
    transform_matrix = torch.cat((transform_matrix, homo_agument), dim=1)

    translation_matrix = torch.as_tensor([[1.0, 0., 0., 0.],
                                          [0.0, 1., 0., 0.],
                                          [0.0, 0., 1., 0.],
                                          [0.0, 0., 0., 1.]]).to(transform_matrix.dtype)

    translation_matrix = translation_matrix.unsqueeze(dim=0).repeat(len(translation), 1, 1)
    translation_matrix[:, 1, -1] = translation
    transform_matrix = torch.einsum('nij,njk->nik', transform_matrix, translation_matrix)
    #transform_matrix = transform_matrix.numpy()
    #print(transform_matrix.shape)

    # flip_trans = torch.as_tensor([[1.0, 0., 0., 0.],
    #                               [0.0, -1., 0., 0.],
    #                               [0.0, 0., -1., 0.],
    #                               [0.0, 0., 0., 1.]])
    # transform_matrix = torch.einsum('nij,jk->nik', transform_matrix, flip_trans)
    return transform_matrix



def orthognal_grasps(geometry_mask, depth_projection, sample_normal, des_normals, sample_pos):

    '''
    :param geometry_mask: [bool,bool,,]
    :param depth_projection:
    :param sample_normal:
    :param des_normals:
    :param sample_pos:
    :return: mX4X4 matrices that used to execute grasp in simulation
    '''
    # if these is no reasonable points do nothing
    assert sum(geometry_mask)>0
    depth = depth_projection[geometry_mask]
    # finger depth
    gripper_dis_from_source = (0.072-0.007 - depth).unsqueeze(dim=-1)
    z_axis = -sample_normal[geometry_mask]  # todo careful
    y_axis = des_normals[geometry_mask]
    x_axis = torch.cross(y_axis, z_axis,dim=1)
    x_axis = F.normalize(x_axis, p=2,dim=1)
    y_axis = torch.cross(z_axis, x_axis,dim=1)
    y_axis = F.normalize(y_axis, p=2, dim=1)
    gripper_position = gripper_dis_from_source.repeat(1, 3) * (-z_axis) + sample_pos[geometry_mask]
    transform_matrix = torch.cat((x_axis.unsqueeze(dim=-1), y_axis.unsqueeze(dim=-1),
                                  z_axis.unsqueeze(dim=-1), gripper_position.unsqueeze(dim=-1)), dim=-1)
    homo_agument = torch.as_tensor([0., 0., 0., 1.]).reshape(1, 1, 4).repeat(len(z_axis), 1, 1).to(des_normals.device)
    transform_matrix = torch.cat((transform_matrix, homo_agument), dim=1)
    #transform_matrix = transform_matrix.numpy()
    #print(transform_matrix.shape)

    # flip_trans = torch.as_tensor([[1.0, 0., 0., 0.],
    #                               [0.0, -1., 0., 0.],
    #                               [0.0, 0., -1., 0.],
    #                               [0.0, 0., 0., 1.]])
    # transform_matrix = torch.einsum('nij,jk->nik', transform_matrix, flip_trans)
    return transform_matrix

def create_gripper_geometry(width=0.08, finger_length=0.05, depth=0.01, finger_width=0.005):
    """
    Creates an Open3D LineSet representation of a parallel gripper.
    Origin is centered between the fingers at the base.
    Z-axis points outwards (approach direction).
    Y-axis points along the finger opening direction.
    X-axis is orthogonal.

    Args:
        width (float): The distance between the open fingers.
        finger_length (float): The length of the fingers.
        depth (float): The depth of the base/fingers (along Z).
        finger_width (float): Thickness of the fingers (along X).

    Returns:
        open3d.geometry.LineSet: The gripper geometry.
    """
    # Define points for the gripper base and fingers
    # Base points
    base_center_back = [0, 0, -depth / 2.0]
    base_center_front = [0, 0, depth / 2.0]

    # Finger 1 (positive Y)
    f1_base_inner = [ -finger_width / 2.0, width / 2.0, -depth / 2.0]
    f1_base_outer = [  finger_width / 2.0, width / 2.0, -depth / 2.0]
    f1_tip_inner  = [ -finger_width / 2.0, width / 2.0, finger_length]
    f1_tip_outer  = [  finger_width / 2.0, width / 2.0, finger_length]

    # Finger 2 (negative Y)
    f2_base_inner = [ -finger_width / 2.0, -width / 2.0, -depth / 2.0]
    f2_base_outer = [  finger_width / 2.0, -width / 2.0, -depth / 2.0]
    f2_tip_inner  = [ -finger_width / 2.0, -width / 2.0, finger_length]
    f2_tip_outer  = [  finger_width / 2.0, -width / 2.0, finger_length]

    # Define points array
    points = [
        # Base representation (optional, connecting finger bases)
        f1_base_inner, f1_base_outer, f2_base_inner, f2_base_outer, # 0-3
        # Finger 1
        f1_tip_inner, f1_tip_outer, # 4-5
        # Finger 2
        f2_tip_inner, f2_tip_outer  # 6-7
    ]

    # Define lines (indices into the points array)
    lines = [
        # Base connections
        [0, 1], [2, 3], [0, 2], [1, 3],
        # Finger 1 Edges
        [0, 4], [1, 5], [4, 5],
        # Finger 2 Edges
        [2, 6], [3, 7], [6, 7]
    ]

    # Define colors for the lines (e.g., red for one finger, blue for the other)
    colors = [
        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],[0.5, 0.5, 0.5], # Base grey
        [1, 0, 0], [1, 0, 0], [1, 0, 0], # Finger 1 red
        [0, 0, 1], [0, 0, 1], [0, 0, 1]  # Finger 2 blue
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # --- Correction based on orthognal_grasps function ---
    # The orthognal_grasps function seems to define:
    # z_axis = -sample_normal (approach direction, seems correct for Z out)
    # y_axis = des_normals (opening direction, seems correct for Y along fingers)
    # x_axis = cross(y, z)
    # However, the gripper model was built with Z pointing *out* from the base towards the fingertips.
    # The transformation matrices place the origin at the 'gripper_position' calculated in the function.
    # Let's flip the gripper's Z-axis and adjust the finger length origin so it matches common conventions
    # where the grasp transform's origin is between the fingertips and Z points *away* from the object.

    # Create a transformation to flip Z and Y and translate origin to fingertips midpoint
    # Flip Z: Rotate 180 degrees around X axis
    flip_transform = np.array([
        [1, 0,  0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0,  0, 1]
    ])
    # Translate origin to be roughly between finger tips (adjust based on `gripper_dis_from_source` logic if needed)
    # Let's assume the calculated 'gripper_position' is the desired origin. The model Z currently points 'out'
    # so the flip aligns it. No further translation needed here if the model origin is correct relative to the transform.

    # Apply the flip
    line_set.transform(flip_transform)


    # Adjust gripper width based on input (this assumes a fixed visual width for now)
    # A more complex version could scale the Y coordinates based on a grasp width parameter.
    # print(f"Using visual gripper width: {width}")

    return line_set

# --- MODIFIED Example Usage ---

sample_number = 10
device = 'cuda'


# 1. Load a point cloud
mesh_path = '/home/seung/Workspaces/Datasets/ACRONYM/meshes/1Shelves/1e3df0ab57e8ca8587f357007f9e75d1_rescaled.obj'
mesh = o3d.io.read_triangle_mesh(mesh_path)
pc = mesh.sample_points_uniformly(5000)
pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30))
pc.orient_normals_consistent_tangent_plane(30) # Optional: orient normals



pos = np.asarray(pc.points)
normals = np.asarray(pc.normals)
pos = torch.from_numpy(pos).to(torch.float32).to(device)
normals = torch.from_numpy(normals).to(torch.float32).to(device)

start_time = time.time()

fps_sample = FarthestSamplerTorch()
_, sample = fps_sample(pos,sample_number)
sample = torch.as_tensor(sample).to(torch.long).reshape(-1).to(device)
sample = torch.unique(sample,sorted=True)


sample_pos = pos[sample, :]
radius_p_batch_index = radius(pos, sample_pos, r=0.05, max_num_neighbors=1024)
radius_p_index = radius_p_batch_index[1, :]
radius_p_batch = radius_p_batch_index[0, :]
sample_pos = torch.cat(
[sample_pos[i, :].repeat((radius_p_batch == i).sum(), 1) for i in range(len(sample))],
dim=0)
sample_copy = sample.clone().unsqueeze(dim=-1)
sample_index = torch.cat(
[sample_copy[i, :].repeat((radius_p_batch == i).sum(), 1) for i in range(len(sample))], dim=0)
edges = torch.cat((sample_index, radius_p_index.unsqueeze(dim=-1)), dim=1)
all_edge_index = torch.arange(0,len(edges)).to(device)
des_pos = pos[radius_p_index, :]
des_normals = normals[radius_p_index, :]
relative_pos = des_pos - sample_pos

print('time taken to sample points:', time.time() - start_time)
start_time = time.time()

relative_pos_normalized = F.normalize(relative_pos, p=2, dim=1)
# set up the record
label_record = []
# edge_sample_index = []
quat_record = []
translation_record = []
# only record approach vectors with a angle mask
x_axis = torch.cross(des_normals, relative_pos_normalized)
x_axis = F.normalize(x_axis, p=2, dim=1)
valid_edge_approach = torch.cross(x_axis, des_normals)
valid_edge_approach = F.normalize(valid_edge_approach, p=2, dim=1)
valid_edge_approach = -valid_edge_approach

print('time taken to generate grasps:', time.time() - start_time)
start_time = time.time()

# print('new approachs',valid_edge_approach.shape)
up_dot_mask = torch.einsum('ik,k->i', valid_edge_approach, torch.tensor([0., 0., 1.]).to(device))
relative_norm = torch.linalg.norm(relative_pos, dim=-1)
# print(relative_norm.size())
depth_proj = -torch.sum(relative_pos * valid_edge_approach, dim=-1)
geometry_mask = torch.logical_and(up_dot_mask > -0.1, relative_norm > 0.003)
geometry_mask = torch.logical_and(relative_norm<0.038,geometry_mask)
depth_proj_mask = torch.logical_and(depth_proj > -0.000, depth_proj < 0.04)
geometry_mask = torch.logical_and(geometry_mask, depth_proj_mask)

print('time taken to hi grasps:', time.time() - start_time)

# draw_grasps2(geometry_mask, depth_proj, valid_edge_approach, des_normals, sample_pos, pos, sample, des=None, scores=None)
pose_candidates = orthognal_grasps(geometry_mask, depth_proj, valid_edge_approach, des_normals,
                                sample_pos)

print('time taken to generate grasps:', time.time() - start_time)

# Ensure pose_candidates is on CPU and is a NumPy array
if isinstance(pose_candidates, torch.Tensor):
    pose_candidates_np = pose_candidates.detach().cpu().numpy()
else:
    pose_candidates_np = np.asarray(pose_candidates) # If it's already numpy or list

# List to hold all geometries to visualize
geometries_to_draw = []

# Add the original point cloud
pc.paint_uniform_color([0.7, 0.7, 0.7]) # Color the point cloud grey
geometries_to_draw.append(pc)

# --- Create and Transform Gripper Models ---
max_grasps_to_show = 10 # Limit the number of grasps shown for clarity
grasps_shown = 0

# Create ONE base gripper model
base_gripper = create_gripper_geometry() # Use default dimensions or adjust

# Iterate through the calculated poses
for i in range(min(len(pose_candidates_np), max_grasps_to_show)):
    grasp_pose_matrix = pose_candidates_np[i]

    # Important: Create a *copy* of the base gripper geometry for each pose
    gripper_vis = o3d.geometry.LineSet(base_gripper) # Efficient copy

    # Apply the transformation
    gripper_vis.transform(grasp_pose_matrix)

    # Add the transformed gripper to our list
    geometries_to_draw.append(gripper_vis)
    grasps_shown += 1

print(f"Visualizing {grasps_shown} grasps.")

# Visualize all geometries
if geometries_to_draw:
    o3d.visualization.draw_geometries(geometries_to_draw,
                                      window_name="Grasp Visualization",
                                      point_show_normal=False) # Normals can clutter the view
else:
    print("No valid geometries to visualize.")
