import os
import numpy as np
import time

import torch
from torch.utils.data import DataLoader
from graspnetAPI import GraspGroup, GraspNetEval


from utils.arguments import cfgs

from dataset.graspnet_dataset import GraspNetDataset, collate_fn
from dataset.graspclutter6d_dataset import GraspClutter6DDataset

from models.economicgrasp import EconomicGrasp, pred_decode

import open3d as o3d
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grasp_toolkit.utils.collision_detector import ModelFreeCollisionDetector

# ------------ GLOBAL CONFIG ------------
if not os.path.exists(cfgs.save_dir):
    os.mkdir(cfgs.save_dir)


# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass



# ------ Testing ------------
def inference(cfgs):
    batch_interval = 20
    stat_dict = {}  # collect statistics
    # set model to eval mode (for bn and dp)
    net.eval()
    tic = time.time()
    for batch_idx, batch_data in enumerate(tqdm(TEST_DATALOADER)):
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            elif 'graph' in key:
                for i in range(len(batch_data[key])):
                    batch_data[key][i] = batch_data[key][i].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)

        # Forward pass
        with torch.no_grad():
            end_points = net(batch_data)
            grasp_preds = pred_decode(end_points)

        # Save results for evaluation
        for i in range(cfgs.batch_size):
            data_idx = batch_idx * cfgs.batch_size + i
            preds = grasp_preds[i].detach().cpu().numpy()
            gg = GraspGroup(preds)

            # collision detection
            if cfgs.collision_thresh > 0:
                cloud, _ = TEST_DATASET.get_data(data_idx, return_raw_cloud=True)
                mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
                collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
                gg = gg[~collision_mask]

            # save grasps
            save_dir = os.path.join(cfgs.save_dir, SCENE_LIST[data_idx], cfgs.camera)
            if cfgs.dataset == 'g1b':
                save_path = os.path.join(save_dir, str(data_idx%256).zfill(4)+'.npy')
            elif cfgs.dataset == 'gc6d':
                save_path = os.path.join(save_dir, str(TEST_DATASET.index_to_info[data_idx][1]).zfill(6)+'.npy')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            gg.save_npy(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # gg.save_npy(save_path)

            # gg = gg.sort_by_score()
            # print(gg.scores)
            # # print('Number of grasps:', gg.__len__() )
            # # if gg.__len__() > 50:
            # #     gg = gg[:50]
            # grippers = gg.to_open3d_geometry_list()
            # cloud = o3d.geometry.PointCloud()
            # pc = batch_data['point_clouds'].detach().cpu().numpy()[i]
            # cloud.points = o3d.utility.Vector3dVector(pc.astype(np.float32))
            # # visualize xyz coord
            # coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            # o3d.visualization.draw_geometries([cloud, *grippers, coord])


        if batch_idx % batch_interval == 0:
            toc = time.time()
            print('Eval batch: %d, time: %fs' % (batch_idx, (toc - tic) / batch_interval))
            tic = time.time()





def evaluate(cfgs):
    if cfgs.dataset == 'g1b':
        ge = GraspNetEval(root=cfgs.dataset_root, camera=cfgs.camera, split=cfgs.split)
        if cfgs.split == 'test_seen':
            res, ap = ge.eval_seen(cfgs.save_dir, proc=cfgs.num_workers)
        elif cfgs.split == 'test_similar':
            res, ap = ge.eval_similar(cfgs.save_dir, proc=cfgs.num_workers)
        elif cfgs.split == 'test_novel':
            res, ap = ge.eval_novel(cfgs.save_dir, proc=cfgs.num_workers)
        elif cfgs.split == 'test':
            res, ap = ge.eval_all(cfgs.save_dir, proc=cfgs.num_workers)
    elif cfgs.dataset == 'gc6d':
        ge = GC6DGraspNetEval(root=cfgs.dataset_root, camera=cfgs.camera, split=cfgs.split)
        res, ap = ge.eval_all(cfgs.save_dir, proc=cfgs.num_workers)
    res = res.transpose(3,0,1,2).reshape(6,-1)
    res = np.mean(res,axis=1)
    print("AP0.4",res[1])
    print("AP0.8",res[3])
    print("AP",np.mean(res))
    # res, ap = ge.eval_all(cfgs.save_dir, proc=cfgs.num_workers)
    save_dir = os.path.join(cfgs.save_dir, 'ap_{}.npy'.format(cfgs.camera))
    np.save(save_dir, res)

if __name__ == '__main__':


    # Create Dataset and Dataloader
    if cfgs.dataset == 'g1b':
        TEST_DATASET = GraspNetDataset(cfgs.dataset_root, split=cfgs.split,
                                    camera=cfgs.camera, num_points=cfgs.num_point, remove_outlier=True, augment=False,
                                    load_label=False)
    elif cfgs.dataset == 'gc6d':
        if cfgs.camera == 'realsense':
            cfgs.camera = 'realsense-d435'
        if cfgs.camera == 'kinect':
            cfgs.camera = 'azure-kinect'
        TEST_DATASET =  GraspClutter6DDataset(cfgs.dataset_root, split='test',
                        camera=cfgs.camera, num_points=cfgs.num_point, remove_outlier=True, augment=False,
                        load_label=False)
    else:
        raise ValueError('Unsupported dataset')
    cfgs.num_workers = 4

    SCENE_LIST = TEST_DATASET.scene_list()
    TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
                                num_workers=2, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)

    # Init the model
    net = EconomicGrasp(seed_feat_dim=512, is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'], strict=False)
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))

    if cfgs.mode == 'infer':
        inference(cfgs)
    elif cfgs.mode == 'eval':
        evaluate(cfgs)
    elif cfgs.mode == 'all':
        inference(cfgs)
        evaluate(cfgs)
    else:
        raise ValueError('Unsupported mode, must be infer, eval or all')
