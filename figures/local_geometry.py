
import random
from graspnetAPI import GraspNet
import os
import time
import numpy as np
import open3d as o3d
from transforms3d.euler import euler2mat, quat2mat
from graspnetAPI.utils.utils import generate_scene_model, generate_scene_pointcloud, generate_views, get_model_grasps, plot_gripper_pro_max, transform_points
from graspnetAPI.utils.rotation import viewpoint_params_to_matrix, batch_viewpoint_params_to_matrix
from graspnetAPI.utils.vis import get_camera_parameters

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import json

def create_beautiful_o3d_figure(point_cloud, point_cloud_crop, gripper, save_path="figure.png",
                                camera_json_path='camera_view.json', # <- JSON 파일 경로를 받을 인자 추가
                                background_color=[1.0, 1.0, 1.0],
                                point_size=5.0, resolution=(1920, 1080)):
    """
    Create publication-quality figure using Open3D's advanced rendering.
    Now supports loading camera view from a JSON file.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=resolution[0], height=resolution[1], visible=False) # Headless rendering
    
    # ... (기존 포인트 클라우드 색상/조명 설정 코드는 동일) ...
    if not point_cloud.has_normals():
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    points = np.asarray(point_cloud.points)
    z_coords = points[:, 2]
    z_normalized = (z_coords - z_coords.min()) / (z_coords.max() - z_coords.min())
    colors = plt.cm.plasma(z_normalized)[:, :3]
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(point_cloud)
    vis.add_geometry(point_cloud_crop)
    gripper.compute_vertex_normals()
    vis.add_geometry(gripper)
    
    render_option = vis.get_render_option()
    render_option.background_color = np.array(background_color)
    render_option.point_size = point_size
    render_option.show_coordinate_frame = False
    render_option.light_on = True
    # render_option.line_width = 100.0
    # --- 뷰 컨트롤 로직 수정 ---
    view_control = vis.get_view_control()

    # camera_json_path가 제공되고 파일이 존재하면, 해당 뷰를 불러옵니다.
    if camera_json_path and os.path.exists(camera_json_path):
        print(f"'{camera_json_path}' 에서 카메라 뷰를 불러옵니다...")
        with open(camera_json_path, 'r') as f:
            cam_params_dict = json.load(f)
        
        # PinholeCameraParameters 객체 생성 및 데이터 채우기
        params = o3d.camera.PinholeCameraParameters()
        params.intrinsic.width = cam_params_dict['intrinsic']['width']
        params.intrinsic.height = cam_params_dict['intrinsic']['height']
        params.intrinsic.intrinsic_matrix = np.array(cam_params_dict['intrinsic']['intrinsic_matrix'])
        params.extrinsic = np.array(cam_params_dict['extrinsic'])
        
        # 뷰 컨트롤에 파라미터 적용
        view_control.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
    
    else:
        # JSON 파일이 없으면 기존의 자동 뷰 설정 방식을 사용합니다.
        print("카메라 JSON 파일이 없어 기본 뷰를 사용합니다.")
        bbox = point_cloud.get_axis_aligned_bounding_box()
        # view_control.fit_in_geometry(bbox) # 간단하게 객체에 뷰를 맞춤
        # view_control.set_zoom(0.7)

    # Render and save
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(save_path, do_render=True)
    
    # remove the point_cloud
    save_path_2 = save_path.replace('.png', '_no_pc.png')
    vis.remove_geometry(point_cloud)
    vis.remove_geometry(gripper)
    params = o3d.camera.PinholeCameraParameters()
    params.intrinsic.width = cam_params_dict['intrinsic']['width']
    params.intrinsic.height = cam_params_dict['intrinsic']['height']
    params.intrinsic.intrinsic_matrix = np.array(cam_params_dict['intrinsic']['intrinsic_matrix'])
    params.extrinsic = np.array(cam_params_dict['extrinsic'])
    
    # 뷰 컨트롤에 파라미터 적용
    view_control.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
    vis.capture_screen_image(save_path_2, do_render=True)
    
    vis.destroy_window()
    
    # read image and crop
    import cv2
    img = cv2.imread(save_path)
    # remove white border
    non_white = np.where(np.all(img < 255, axis=-1))
    y1, y2 = np.min(non_white[0]), np.max(non_white[0])
    x1, x2 = np.min(non_white[1]), np.max(non_white[1])
    margin = 300
    y1 = max(0, y1-margin)
    y2 = min(img.shape[0], y2+margin)
    x1 = max(0, x1-margin)
    x2 = min(img.shape[1], x2+margin)
    img_cropped = img[y1:y2, x1:x2]
    cv2.imwrite(save_path, img_cropped)
    print(f"✅ Figure saved to '{save_path}'")
    
    img = cv2.imread(save_path_2)
    # remove white border
    non_white = np.where(np.all(img < 255, axis=-1))
    y1, y2 = np.min(non_white[0]), np.max(non_white[0])
    x1, x2 = np.min(non_white[1]), np.max(non_white[1])
    margin = 300
    y1 = max(0, y1-margin)
    y2 = min(img.shape[0], y2+margin)
    x1 = max(0, x1-margin)
    x2 = min(img.shape[1], x2+margin)
    img_cropped = img[y1:y2, x1:x2]
    cv2.imwrite(save_path_2, img_cropped)
    print(f"✅ Figure without point cloud saved to '{save_path_2}'")
    

    return save_path


def enhance_your_visualization(obj_cloud, cropped_cloud, gripper, obj_idx):
    """
    Enhanced version of your existing visualization code
    """
    # Create enhanced point cloud
    model = o3d.geometry.PointCloud()
    model.points = o3d.utility.Vector3dVector(obj_cloud)
    
    # Estimate normals for better lighting
    model.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    # Apply sophisticated coloring
    points = np.asarray(model.points)
    
    # Create color based on surface curvature or height
    if model.has_normals():
        normals = np.asarray(model.normals)
        # Color based on normal direction for surface features
        colors = (normals + 1) / 2  # Normalize to [0,1]
    else:
        # Fallback to height-based coloring
        z_coords = points[:, 2]
        z_normalized = (z_coords - z_coords.min()) / (z_coords.max() - z_coords.min())
        colors = plt.cm.viridis(z_normalized)[:, :3]
    
    model.colors = o3d.utility.Vector3dVector(colors)
    
    model_crop = o3d.geometry.PointCloud()
    model_crop.points = o3d.utility.Vector3dVector(cropped_cloud.points)
    model_crop.paint_uniform_color([0, 1, 0])  # Green for cropped region
    model_crop.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    # Create multiple beautiful renderings
    create_beautiful_o3d_figure(model, model_crop, gripper, f"object_{obj_idx}_beautiful.png")
    
    print(f"Created beautiful visualizations for object {obj_idx}")


def save_view_point(pcd, cropped_pcd, grippers, filename="camera_view.json"):
    """
    Open3D의 인터랙티브 창에서 뷰포인트를 설정하고 저장하는 함수
    
    사용법:
    1. 창이 뜨면 마우스로 원하는 뷰를 자유롭게 설정하세요.
    2. 원하는 뷰가 완성되면 키보드에서 'S' 키를 누르세요.
    3. 뷰포인트가 파일로 저장되며, 창은 그대로 유지됩니다.
    4. 창을 닫으려면 'Q'를 누르거나 창의 X 버튼을 클릭하세요.
    """
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(pcd)
    # set transparency to grippers
    grippers.paint_uniform_color([1, 0, 0])
    grippers.compute_vertex_normals()
    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = True
    render_option.background_color = np.array([1, 1, 1])
    render_option.point_size = 5.0
    render_option.light_on = True
    render_option.mesh_show_wireframe = False
    render_option.mesh_show_back_face = True
    render_option.line_width = 100.0
    vis.add_geometry(cropped_pcd)
    vis.add_geometry(gripper)

    def save_params(visualizer):
        params = visualizer.get_view_control().convert_to_pinhole_camera_parameters()
        
        with open(filename, 'w') as f:
            cam_params_dict = {
                'intrinsic': {
                    'width': params.intrinsic.width,
                    'height': params.intrinsic.height,
                    'intrinsic_matrix': np.asarray(params.intrinsic.intrinsic_matrix).tolist()
                },
                'extrinsic': np.asarray(params.extrinsic).tolist()
            }
            json.dump(cam_params_dict, f, indent=4)
            
        print(f"✅ 카메라 뷰포인트가 '{filename}'에 저장되었습니다. (창은 유지됩니다)")
        
        # --- 이 부분이 중요합니다 ---
        # visualizer.destroy_window() # 이 줄을 주석 처리하거나 삭제합니다.
        
        return False

    vis.register_key_callback(ord("S"), save_params)
    
    print("✨ 뷰를 조정한 후 'S' 키를 눌러 저장하세요. 'Q' 키를 누르면 창이 닫힙니다.")
    vis.run()



# Usage example with your data:
# enhance_your_visualization(obj_cloud, grippers, obj_idx)

dataset_root = '/home/seung/Datasets/GraspNet-1Billion'  # path to GraspNet-1Billion

split = 'test_novel'
# initialize a GraspNet instance  
g = GraspNet(dataset_root, camera='kinect', split=split)
obj_ids = g.getObjIds(g.getSceneIds())
th = 0.5
max_width = 0.08
num_grasp = 1

# random.shuffle(obj_ids)
print(len(obj_ids), 'objects to process')
# show object grasps
for obj_idx in obj_ids:
    
    plyfile = os.path.join(dataset_root, 'models', '%03d'%obj_idx, 'nontextured.ply')
    model = o3d.io.read_point_cloud(plyfile)
    model.paint_uniform_color([0.5, 0.5, 0.5])

    num_views, num_angles, num_depths = 300, 12, 4
    views = generate_views(num_views)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width = 1280, height = 720)
    ctr = vis.get_view_control()
    param = get_camera_parameters(camera='kinect')

    cam_pos = np.load(os.path.join(dataset_root, 'scenes', 'scene_0000', 'kinect', 'cam0_wrt_table.npy'))
    param.extrinsic = np.linalg.inv(cam_pos).tolist()

    sampled_points, offsets, scores, _ = get_model_grasps('%s/grasp_label/%03d_labels.npz'%(dataset_root, obj_idx))

    cnt = 0
    point_inds = np.arange(sampled_points.shape[0])
    np.random.shuffle(point_inds)
    grippers = []

    for point_ind in point_inds:
        target_point = sampled_points[point_ind]
        offset = offsets[point_ind]
        score = scores[point_ind]
        view_inds = np.arange(300)
        np.random.shuffle(view_inds)
        flag = False
        for v in view_inds:
            if flag: break
            view = views[v]
            angle_inds = np.arange(12)
            np.random.shuffle(angle_inds)
            for a in angle_inds:
                if flag: break
                depth_inds = np.arange(4)
                d = 1
                # for d in depth_inds:
                if flag: break
                angle, depth, width = offset[v, a, d]
                if score[v, a, d] > th or score[v, a, d] < 0 or width > max_width:
                    continue
                R = viewpoint_params_to_matrix(-view, angle)
                t = target_point
                # gripper = plot_gripper_pro_max(t, R, width, depth, 1.1-score[v, a, d])
                t_ = np.zeros(3)
                R_ = np.eye(3)
                gripper = plot_gripper_pro_max(t_, R_, width, depth, 1.0)
                grippers.append(gripper)
                
                flag = True
                    
        if flag:
            cnt += 1
        if cnt == num_grasp:
            grasps = np.array([[score, width, depth, *R.reshape(-1), *t, 0]])
            break
    # colorize green the inside gripper region
    grasp_points = grasps[:, 13:16]
    grasp_poses = grasps[:, 4:13].reshape([-1,3,3])
    grasp_depths = grasps[:, 3]
    grasp_widths = grasps[:, 1]
    
    obj_cloud = np.asarray(model.points)
    transform = np.eye(4)
    transform[:3,:3] = R
    transform[:3,3] = t
    transform = np.linalg.inv(transform)
    ones = np.ones(obj_cloud.shape[0])[:, np.newaxis]
    cloud_ = np.concatenate([obj_cloud, ones], axis=1)
    cloud_transformed = np.dot(transform, cloud_.T).T
    cloud_transformed = cloud_transformed[:, :3]
    obj_cloud = cloud_transformed
    
    model = o3d.geometry.PointCloud()
    model.points = o3d.utility.Vector3dVector(obj_cloud)
    model.paint_uniform_color([0.5,0.5,0.5])
    
    
    # target = (obj_cloud[np.newaxis,:,:] - grasp_points[:,np.newaxis,:])
    # target = np.matmul(target, grasp_poses)
    target = obj_cloud[np.newaxis,:,:] 
# 
    ## crop the object in gripper closing area
    height = 0.02
    depth_base = 0.02
    depth_outer = 0.03
    mask1 = ((target[:,:,2]>-height/2) & (target[:,:,2]<height/2))
    mask2 = ((target[:,:,0]>-depth_base) & (target[:,:,0]< depth_outer)) #grasp_depths[:,np.newaxis] + depth_outer))
    mask4 = (target[:,:,1]<-grasp_widths[:,np.newaxis]/2)
    mask6 = (target[:,:,1]>grasp_widths[:,np.newaxis]/2)
    inner_mask = (mask1 & mask2 &(~mask4) & (~mask6)) # [n_batch, n_points]
    obj_cloud_inner = obj_cloud[inner_mask[0]]
    cropped_cloud = o3d.geometry.PointCloud()
    cropped_cloud.points = o3d.utility.Vector3dVector(obj_cloud_inner)
    cropped_cloud.paint_uniform_color([0,1,0])
    


    # filename = os.path.join(save_folder, 'object_{}_grasp.png'.format(obj_idx))
    # vis.capture_screen_image(filename, do_render=True)
    # o3d.visualization.draw_geometries([model, *grippers])
    
    # save_view_point(model, cropped_cloud, gripper, filename="camera_view.json")
    enhance_your_visualization(obj_cloud, cropped_cloud, gripper, obj_idx)


