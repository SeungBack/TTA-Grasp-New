import h5py
import open3d as o3d
import glob
import numpy as np

input_path = '/home/seung/Workspaces/Datasets/GraspNet-1Billion/grasp_qnet/kinect/train/scene_0000_000000_0001_0.7_14826371.h5'
# input_dir = '/SSDg/shback/graspnet_1billion/gs_dataset/kinect/train'
# input_paths = sorted(glob.glob(input_dir + '/*.h5'))

# read h5 file
# scores = []
# for input_path in input_paths:
f = h5py.File(input_path, 'r')

# read data
obj_cloud = f['obj_cloud'][()] 
gripper_cloud = f['gripper_cloud'][()] 
score = f['score'][()] 
scene_id = f['scene_id'][()]
img_id = f['img_id'][()]
obj_id = f['obj_id'][()]
f.close()
# print(obj_cloud.shape, gripper_cloud.shape, float(score), str(scene_id), str(img_id), str(obj_id))
# scores.append(score)
# print(np.unique(scores))

obj_cloud_inner_pcd = o3d.geometry.PointCloud()
obj_cloud_inner_pcd.points = o3d.utility.Vector3dVector(obj_cloud)
obj_cloud_inner_pcd.paint_uniform_color([0, 1, 0]) # green
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
gripper_pcd = o3d.geometry.PointCloud()
gripper_pcd.points = o3d.utility.Vector3dVector(gripper_cloud)
gripper_pcd.paint_uniform_color([1, 0, 0]) # red
o3d.visualization.draw_geometries([obj_cloud_inner_pcd, gripper_pcd, coord_frame])