import sys
import os
import argparse
import time
import torch
import wandb  # Import wandb
import numpy as np
import open3d as o3d

from tqdm import tqdm
from omegaconf import OmegaConf
from graspnetAPI import GraspGroup, GraspNetEval
from torch.utils.data import DataLoader

from models.graspnet import load_graspnet
from dataset.graspnet_dataset import GraspNetDataset, collate_fn
from dataset.graspclutter6d_dataset import GraspClutter6DDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grasp_toolkit.tta import get_tta_method
from grasp_toolkit.utils import ModelFreeCollisionDetector, print_cfg, init_wandb, set_seed


def get_dataset(cfg, dataset_name, split, camera):
    if dataset_name == 'graspnet1b':
        # Init datasets and dataloaders
        test_dataset = GraspNetDataset(
            cfg.graspnet1b.root, 
            valid_obj_idxs=None, 
            grasp_labels=None, 
            split=split,
            camera=camera, 
            num_points=cfg.model.num_point, 
            remove_outlier=True, 
            augment=False, 
            load_label=False, 
            return_raw_cloud=cfg.tta.method in ['cotta', 'tta-grasp']
        )
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=cfg.tta.batch_size, 
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn
        )
    elif dataset_name == 'graspclutter6d':
        test_dataset = GraspClutter6DDataset(
            cfg.graspclutter6d.root, 
            valid_obj_idxs=None, 
            grasp_labels=None, 
            split=split,
            camera=camera, 
            num_points=cfg.model.num_point, 
            remove_outlier=True, 
            augment=False, 
            load_label=False, 
            return_raw_cloud=cfg.tta.method in ['cotta', 'tta-grasp']
        )
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=cfg.tta.batch_size, 
            shuffle=False,
            num_workers=cfg.num_workers, 
            collate_fn=collate_fn
        )
    else:
        raise ValueError(f'Invalid dataset: {dataset_name}')
    return test_dataset, test_dataloader


def inference(cfg, graspnet, dataset_name, split, camera):
    set_seed(cfg.seed)
    """Run inference on the dataset."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    test_dataset, test_dataloader = get_dataset(cfg, dataset_name, split, camera)
    tta_method = get_tta_method(cfg, graspnet)

    times = []
    for batch_idx, batch_data in enumerate(tqdm(test_dataloader)):
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)

        # Forward pass
        tic = time.time()
        grasp_preds, end_points = tta_method.forward(batch_data)
        toc = time.time()
        batch_time = toc - tic
        times.append(batch_time)
        if batch_idx % 5000 == 0 and batch_idx != 0:
            tta_method.save_model(iter=batch_idx)

        if cfg.use_wandb:
            wandb.log({
                "tta_time": batch_time / cfg.tta.batch_size,
            })
            if end_points is not None:
                for k in ['loss/view_loss', 'loss/grasp_loss', 'loss/overall_loss', 'loss/tent_loss', 'loss/contrastive_loss']:
                    if k in end_points:
                        wandb.log({k: end_points[k].item()})
        end_points = None
        # Save results
        scene_list = test_dataset.scene_list()
        anno_list = test_dataset.frameid
        for i in range(batch_data['point_clouds'].shape[0]):
            # Set random seed
            data_idx = batch_idx * cfg.tta.batch_size + i
            if isinstance(grasp_preds[i], torch.Tensor):
                gg = GraspGroup(grasp_preds[i].detach().cpu().numpy())
            else:
                gg = GraspGroup(grasp_preds[i])
            # Apply collision detection if needed
            if cfg.model.collision_thresh > 0:
                cloud = test_dataset.get_data(data_idx, return_raw_cloud=True)['point_clouds_raw']
                mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfg.model.voxel_size)
                collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfg.model.collision_thresh)
                gg = gg[~collision_mask]

                # visualize the gg and cloud
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(cloud)
                # gg= gg.nms()
                # gg = gg.sort_by_score()[:50]
                # o3d.visualization.draw_geometries([pcd] + gg.to_open3d_geometry_list())
            
            # Save grasps
            save_dir = os.path.join(cfg.dump_dir, scene_list[data_idx], camera)
            save_path = os.path.join(save_dir, str(anno_list[data_idx]).zfill(4)+'.npy')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            gg.save_npy(save_path)
    tta_method.save_model()
    avg_time = np.mean(times) / cfg.tta.batch_size
    print(f'Average TTA time: {avg_time:.2f}s')
    if cfg.use_wandb:
        wandb.log({
            "avg_inference_time": avg_time,
        })

def evaluate(cfg):
    test_dataset, test_dataloader = get_dataset(cfg, cfg.eval_dataset.name, cfg.eval_dataset.split, cfg.eval_dataset.camera)
    # Set random seed
    set_seed(cfg.seed)
    """Evaluate the results using GraspNetEval."""
    if 'mini' in cfg.eval_dataset.split:
        # remove all previous results
        for scene_id, img_id in test_dataset.scene_id_img_id_pairs:
            acc_list_path = os.path.join(cfg.dump_dir, 'scene_{:04d}'.format(scene_id), cfg.eval_dataset.camera, '{:04d}_acc.npy'.format(img_id))
            if os.path.exists(acc_list_path):
                os.remove(acc_list_path)
        ge = GraspNetEval(root=cfg.graspnet1b.root, camera=cfg.eval_dataset.camera, split='custom')
        scene_id = np.unique([int(x.split('_')[-1]) for x in test_dataset.scene_list()])
        res = ge.parallel_eval_scenes(scene_id, cfg.dump_dir, 
                                               test_dataset.img_id_per_scene, 
                                               proc=cfg.num_workers)
        res = np.array(res)
    else:
        cfg.eval_dataset.split = cfg.eval_dataset.split.replace('_mixed', '')
        ge = GraspNetEval(root=cfg.graspnet1b.root, camera=cfg.eval_dataset.camera, split=cfg.eval_dataset.split)
        scene_id = np.unique([int(x.split('_')[-1]) for x in test_dataset.scene_list()])
        res = np.array(ge.parallel_eval_scenes(scene_id, cfg.dump_dir, proc=cfg.num_workers))

    save_dir = os.path.join(cfg.dump_dir, f'ap_{cfg.eval_dataset.camera}.npy')
    np.save(save_dir, res)

    # # #!TODO: fix this
    res = []
    for scene_id, img_id in test_dataset.scene_id_img_id_pairs:
        acc_list_path = os.path.join(cfg.dump_dir, 'scene_{:04d}'.format(scene_id), cfg.eval_dataset.camera, '{:04d}_acc.npy'.format(img_id))
        if not os.path.exists(acc_list_path):
            print(f'{acc_list_path} does not exist')
            acc = np.zeros((6))
        else:
            acc = np.load(acc_list_path)
        res.append(acc)
    res = np.array(res).transpose(2, 0, 1).reshape(6, -1)

    res_mean = np.mean(res, axis=1)
    ap_04 = res_mean[1]*100
    ap_08 = res_mean[3]*100
    ap_mean = np.mean(res_mean)*100
    metrics = {
        f"{cfg.eval_dataset.split}_AP_0.4": ap_04,
        f"{cfg.eval_dataset.split}_AP_0.8": ap_08,
        f"{cfg.eval_dataset.split}_AP_mean": ap_mean
    }
    if cfg.use_wandb:
        wandb.log(metrics)
    # Print evaluation results
    print('\n')
    print('------------------- Evaluation Results -------------------')
    print(f'Split: {cfg.eval_dataset.split}')
    print(f"AP 0.4: {ap_04:.2f}")
    print(f"AP 0.8: {ap_08:.2f}")
    print(f'AP    : {ap_mean:.2f}')
    print(f'{ap_04:.2f}, {ap_08:.2f}, {ap_mean:.2f}')
    return metrics

def test(cfg):
    """Main testing function."""
    # Create output directory
    if not os.path.exists(cfg.dump_dir): 
        os.makedirs(cfg.dump_dir)

    if 'infer' in cfg.mode:
        # Initialize device and model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        graspnet = load_graspnet(cfg, device)
        if 'graspclutter6d' == cfg.tta_dataset.name:
            # inference(cfg, graspnet, 'graspclutter6d', cfg.tta_dataset.split, cfg.tta_dataset.camera)
            # after adapting, run inference on graspnet1b for evaluation
            cfg.tta.method = 'notta'
            cfg.model.ckpt_path = os.path.join(cfg.dump_dir, 'checkpoint.pth')
            print(f'cfg.model.ckpt_path: {cfg.model.ckpt_path}')
            graspnet = load_graspnet(cfg, device)
            inference(cfg, graspnet, 'graspnet1b', cfg.eval_dataset.split, cfg.eval_dataset.camera)

        if 'graspnet1b' == cfg.tta_dataset.name:
            inference(cfg, graspnet, 'graspnet1b', cfg.tta_dataset.split, cfg.tta_dataset.camera)
        graspnet = None # clear memory

    if 'eval' in cfg.mode:
        metrics = evaluate(cfg)
        # Finish the run with summary metrics
    if cfg.use_wandb:
        for key, value in metrics.items():
            wandb.run.summary[key] = value

def main(cfg, args):
    """Main function for normal execution."""
    # Add experiment name if not present
    if not hasattr(cfg, 'exp_name'):
        cfg.exp_name = f"{cfg.dump_dir.replace('/', '_')}"
    # Print configuration
    print_cfg(cfg)
    
    # Initialize wandb only if not in debug mode
    if args.wandb:
        init_wandb(cfg)
        
    try:
        # Run test
        test(cfg)
    finally:
        # Finish wandb run if it was initialized
        if not args.debug and wandb.run is not None:
            wandb.finish()

def sweep_agent(sweep_cfg):
    """Function to be called by wandb agent."""
    # Parse the original arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=None, help='Path to cfg file')
    parser.add_argument('--opts', nargs='+', default=[], help='Override configuration options')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--sweep', action='store_true', help='Run in sweep mode')
    args = parser.parse_args()
    
    # Skip wandb initialization in debug mode
    if args.debug:
        print("Debug mode active, skipping wandb initialization")
        return
    
    wandb.init()
   
    cfg = OmegaConf.load(args.cfg)
    base_cfg = OmegaConf.load(cfg.base_cfg)
    cfg = OmegaConf.merge(base_cfg, cfg)
    cfg.use_wandb = True
     # Load the base configuration
    if 'dump_dir' not in cfg.keys() or cfg.dump_dir == '':
        cfg.dump_dir = args.cfg.replace('configs', 'logs').replace('.yaml', '')
    
    # Update config with wandb sweep values
    cfg.tta.contrastive_loss.thresh = wandb.config.thresh
    cfg.tta.contrastive_loss.use_off_object = wandb.config.use_off_object
    cfg.tta.contrastive_loss.temperature = wandb.config.temperature
    cfg.tta.contrastive_loss.weight = wandb.config.weight
    cfg.tta.contrastive_loss.n_proj_layers = wandb.config.n_proj_layers
    cfg.tta.contrastive_loss.proj_dim = wandb.config.proj_dim
    cfg.tta.contrastive_loss.use_bn = wandb.config.use_bn
    # Merge additional options if provided
    if args.opts:
        cfg.merge_with_dotlist(args.opts)
    
    # Set random seed
    set_seed(cfg.seed)
    
    # Print configuration
    print_cfg(cfg)
    
    try:
        # Run test with the sweep configuration
        test(cfg)
    finally:
        # Make sure to finish the run
        if wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=None, help='Path to cfg file')
    parser.add_argument('--opts', nargs='+', default=[], help='Override configuration options')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--sweep', action='store_true', help='Run in sweep mode')
    parser.add_argument('--wandb', action='store_true', help='Run in wandb mode')
    args = parser.parse_args()
    
    if args.debug:
        print("Debug mode active, wandb logging will be disabled")
    
    # Load configuration
    cfg = OmegaConf.load(args.cfg)
    
    if 'dump_dir' not in cfg.keys() or cfg.dump_dir == '':
        cfg.dump_dir = args.cfg.replace('configs', 'logs').replace('.yaml', '')
    print('dump_dir: ', cfg.dump_dir)
    base_cfg = OmegaConf.load(cfg.base_cfg)
    cfg = OmegaConf.merge(base_cfg, cfg)
    cfg.use_wandb = args.wandb

    # Convert opts to a list if it's not empty
    if args.opts:
        print(args.opts)  # This will now be a list directly
        cfg.merge_with_dotlist(args.opts)

    # Define the sweep configuration
    sweep_configuration = {
        "method": "bayes",  # "bayes", "grid"
        "metric": {"goal": "maximize", "name": "{}_AP_mean".format(cfg.eval_dataset.split)},
        "parameters": {
            "thresh": {"values": [0.3, 0.4, 0.5]},
            "use_off_object": {"values": [True, False]},
            "temperature": {"values": [0.05, 0.07, 0.1, 0.15, 0.2]},
            "weight": {"values": [0.001, 0.0005, 0.0001]},
            "n_proj_layers": {"values": [2, 3]},
            "proj_dim": {"values": [64, 128, 256]},
            "use_bn": {"values": [True, False]},
            "p": {"values": [2]},
            
        }
    }
    
    # Check if we're running a sweep or a regular run
    if args.sweep:
        # Skip wandb in debug mode
            # Initialize the sweep
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="Grasp-TTA-New")
        # Start the sweep agent
        wandb.agent(sweep_id, function=lambda: sweep_agent(cfg), count=300)
    else:
        # Run the normal main function
        main(cfg, args)