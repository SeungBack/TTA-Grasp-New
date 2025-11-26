

class EconomicGrasp:
    def __init__(self, cfg):
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = load_economicgrasp(cfg, device)
        self.tta_method = get_tta_method(cfg, model)

    def __call__(self, state):

        extrinsic = np.array([[ 1.00000000e+00,  6.12323400e-17,  3.08148791e-33, -1.50000000e-01],
                    [ 3.06161700e-17, -5.00000000e-01, -8.66025404e-01,  1.61602540e-01],
                    [-5.30287619e-17,  8.66025404e-01, -5.00000000e-01,  5.20096189e-01],
                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        pcd_o3d = state.pc
        pcd_o3d = pcd_o3d.transform(extrinsic)
        pcd = np.asarray(pcd_o3d.points)
        

            
        g_array = np.array(g_array)

        gg = GraspGroup(g_array)

        # grippers = gg.to_open3d_geometry_list()
        # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        # grippers.append(origin)
        # o3d.visualization.draw_geometries([pcd_o3d, *grippers])

        if gg.__len__() == 0:
            print('no valid grasps')
            return [], [], 0
        if args.collision_thresh > 0:
            # add plane (z=0.055) to the point cloud from -1 to 1
            mfcdetector = ModelFreeCollisionDetector(pcd, voxel_size=0.01)
            collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=args.collision_thresh)
            gg = gg[~collision_mask]

        if gg.__len__() == 0:
            print('no valid grasps')
            return [], [], 0
        
        # gg = gg.nms()
        gg = gg.sort_by_score()
        if gg.__len__() > 10:
            gg = gg[:10]
        
        # grippers = gg.to_open3d_geometry_list()
        # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        # grippers.append(origin)
        # o3d.visualization.draw_geometries([pcd_o3d, *grippers])

        best_g = gg[0]
        best_g = best_g.transform(np.linalg.inv(extrinsic))

        width = best_g.width * 1.2
        width = max(0.08, width)
        score = best_g.score
        trans = best_g.translation
        rot = best_g.rotation_matrix
        # transform -90 degree around y axis
        # turn on for gc6d, g1b, turn off for acrnoym
        rot_ = R.from_euler('y', 90, degrees=True).as_matrix()
        rot = rot @ rot_

        # convert to scipy rotation
        pose = Transform(R.from_matrix(rot), trans)
        grasp = G(pose, width)

        planning_time = 0.01
        return [grasp], [score], planning_time

class EconomicGraspPlanner:
    def __init__(self, args):
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = load_economicgrasp(cfg, device)

    def __call__(self, state):

        extrinsic = np.array([[ 1.00000000e+00,  6.12323400e-17,  3.08148791e-33, -1.50000000e-01],
                    [ 3.06161700e-17, -5.00000000e-01, -8.66025404e-01,  1.61602540e-01],
                    [-5.30287619e-17,  8.66025404e-01, -5.00000000e-01,  5.20096189e-01],
                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        pcd_o3d = state.pc
        pcd_o3d = pcd_o3d.transform(extrinsic)
        pcd = np.asarray(pcd_o3d.points, dtype=np.float32)
        rgb_img = state.rgb
        depth_img = state.depth
        w, h, fx, fy, cx, cy = 1920, 1080, 1415.0, 1424.0, 969.0, 548.0
        # import matplotlib.pyplot as plt
        # plt.imshow(rgb_img)
        # plt.show()
        
        xmap, ymap = np.arange(depth_img.shape[1]), np.arange(depth_img.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depth_img 
        points_x = (xmap - cx) / fx * points_z
        points_y = (ymap - cy) / fy * points_z
        
        mask = (points_z > 0)
        points = np.stack([points_x, points_y, points_z], axis=-1)
        points = points[mask].astype(np.float32)
        colors = rgb_img[mask].astype(np.float32) / 255.0
        # print(points.min(axis=0), points.max(axis=0))
        # print(np.unique(points[:, 2]))
        # print(points.shape, colors.shape)
        # visualize it using open3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        # pcd.colors = o3d.utility.Vector3dVector(colors)
        # o3d.visualization.draw_geometries([pcd])
        
        gg, cloud = self.anygrasp.get_grasp(points, colors, apply_object_mask=True, dense_grasp=False, collision_detection=False)
        pred_line_sets = []
        gg = gg.sort_by_score()
        gg = gg[:200]
        
        g_array = []
        lower = [0.03, 0.03, 0.045]
        upper = [0.27, 0.27, 0.5]
        
        for i in range(len(gg)):
            width = gg[i].width
            if width < 0.001 or width > 0.08:
                continue
            score = gg[i].score
            rot = gg[i].rotation_matrix
            trans = gg[i].translation

            x, y, z = trans.copy()
            _trans = np.array([x, y, z, 1])
            _trans = np.matmul(np.linalg.inv(extrinsic), _trans)
            x, y, z, _ = _trans

            if x < lower[0] or x > upper[0] or y < lower[1] or y > upper[1] or z < lower[2] or z > upper[2]:
                continue
        
            rot = rot.reshape(-1)
            g_array.append([score, width, 0.02, 0.02, *rot, *trans, -1])

            
        g_array = np.array(g_array)

        gg = GraspGroup(g_array)
        if args.collision_thresh > 0:
            try:
                mfcdetector = ModelFreeCollisionDetector(pcd, voxel_size=0.01)
                collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=args.collision_thresh)
            except:
                return [], [], 0
            gg = gg[~collision_mask]
        if gg.__len__() == 0:
            return [], [], 0
        
        # gg = gg.nms()
        gg = gg.sort_by_score()
        if gg.__len__() > 10:
            gg = gg[:10]
        
        # grippers = gg.to_open3d_geometry_list()
        # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        # grippers.append(origin)
        # o3d.visualization.draw_geometries([pcd_o3d, *grippers])

        best_g = gg[0]
        best_g = best_g.transform(np.linalg.inv(extrinsic))

        width = best_g.width * 1.2
        width = max(0.08, width)
        score = best_g.score
        trans = best_g.translation
        rot = best_g.rotation_matrix
        # turn on for gc6d, g1b, turn off for acrnoym
        rot_ = R.from_euler('y', 90, degrees=True).as_matrix()
        rot = rot @ rot_

        # convert to scipy rotation
        pose = Transform(R.from_matrix(rot), trans)
        grasp = G(pose, width)

        planning_time = 0.01
        return [grasp], [score], planning_time
