import open3d as o3d
import numpy as np
import glob
import os
import random

root = '/home/seung/Workspaces/Datasets/GraspAnything6D/pc'
paths = glob.glob(root + '/*.npy')
for path in random.sample(paths, 10):
    data = np.load(path)
    print(data.shape)
    points = data[:, :3]
    colors = data[:, 3:]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
