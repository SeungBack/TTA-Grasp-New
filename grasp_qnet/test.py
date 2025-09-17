import os
import sys
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn
# Import the dataset class (assuming the same import from original code)
from dataset.data_loader import GraspEvalDataset

# Import the models
from models.pointnet_v2 import *
from models.dgcnn import *
from sklearn.metrics import ndcg_score
from models.edgegrasp import *
import time
import random

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, '..'))


def log_string(out_str, log_dir=None):
    """Print and optionally log a string."""
    if log_dir:
        with open(os.path.join(log_dir, 'log_inference.txt'), 'a') as log_fout:
            log_fout.write(out_str + '\n')
            log_fout.flush()
    print(out_str)


def run_inference(cfgs):
    """Run inference on the test data."""
    seed = 7
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    
    # Create output directories
    output_dir = cfgs.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create visualization directory
    if cfgs.visualize:
        vis_dir = os.path.join(output_dir, 'visualizations')
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
    
    test_dataset = GraspEvalDataset(cfgs.g1b_root, cfgs.acronym_root, split=cfgs.test_split, return_score=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if cfgs.net == 'pointnet2':
        net = PointNet2GraspQNet()
    elif cfgs.net == 'dgcnn':
        net = DGCNNGraspQNet()
    elif cfgs.net == 'edgegrasp':
        net = EdgeGraspQNet()
    else:
        raise ValueError('Network not supported')

    # Load model to GPU(s)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    net = net.to(device)

    checkpoint = torch.load(cfgs.ckpt_path, map_location=device)
    
    try:
        # Try different ways to load the model state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            # If checkpoint is just the state dict itself
            state_dict = checkpoint
        
        # Handle DataParallel case
        if list(state_dict.keys())[0].startswith('module.') and not isinstance(net, torch.nn.DataParallel):
            # Remove 'module.' prefix for non-DataParallel model
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
                new_state_dict[name] = v
            net.load_state_dict(new_state_dict)
        else:
            # Direct load
            net.load_state_dict(state_dict)
        
    except Exception as e:
        raise


    # Print model parameter count
    num_params = sum(p.numel() for p in net.parameters())
    print(f'Number of parameters: {num_params}')

    # Run inference
    net.eval()  # Set the model to evaluation mode
    
    if cfgs.mc_dropout == True:
        net.initialize_mc_dropout()
    
    # Initialize statistics
    all_pred_cls = []
    all_gt_cls = []
    all_pred_scores = []
    all_gt_scores = []
    all_std_devs = []
    
    threshold = 0.3
    fwd_time = 0.0
    
    with torch.no_grad():  # Disable gradient computation
        for batch_idx, (obj_cloud, gripper_cloud, gt_score) in enumerate(tqdm(test_dataloader)):
            # Move data to device
            tic = time.time()
            obj_cloud = obj_cloud.to(device)
            gripper_cloud = gripper_cloud.to(device)
            
            # Forward pass
            if cfgs.mc_dropout:
                pred, std = net.forward_mc_dropout(obj_cloud, gripper_cloud, N=5)
                all_std_devs.extend(std.cpu().numpy())
                # if std > 0.05:
                #     continue
            else:
                pred = net(obj_cloud, gripper_cloud)
            # print(pred)
            fwd_time += time.time() - tic
                
            pred_label = pred >= threshold
            gt_label = gt_score >= threshold
            all_pred_cls.extend(pred_label.cpu().numpy().astype(int))
            all_gt_cls.extend(gt_label.cpu().numpy().astype(int))
            all_pred_scores.extend(pred.cpu().numpy())
            all_gt_scores.extend(gt_score.cpu().numpy())

            
            
    all_pred_cls = np.array(all_pred_cls).reshape(-1)
    all_gt_cls = np.array(all_gt_cls).reshape(-1)
    all_pred_scores = np.array(all_pred_scores).reshape(-1)
    all_gt_scores = np.array(all_gt_scores).reshape(-1)
    if cfgs.mc_dropout:
        all_std_devs = np.array(all_std_devs).reshape(-1)
            
    fwd_time = fwd_time / len(test_dataloader)
    print(f"Forward time: {fwd_time:.4f} seconds")
    all_gt_cls = np.array(all_gt_cls)
    all_pred_cls = np.array(all_pred_cls)
    all_pred_scores = np.array(all_pred_scores)
    all_gt_scores = np.array(all_gt_scores)
    acc = sklearn.metrics.accuracy_score(all_gt_cls, all_pred_cls)
    print(f"Accuracy: {acc:.4f}")
    mae = np.mean(np.abs(all_gt_scores - all_pred_scores))
    print(f"MAE: {mae:.4f}")
    
    abs_diff = np.abs(all_gt_scores - all_pred_scores)
    # plot histogram of absolute differences
    plt.figure(figsize=(10, 6), dpi=300)
    plt.hist(abs_diff, bins=50, color='blue', alpha=0.7, edgecolor='black')
    plt.title('Histogram of Absolute Differences between Predicted and Ground Truth Scores')
    plt.xlabel('Absolute Difference')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(os.path.join(output_dir, 'abs_diff_histogram.png'))
    plt.close()
    
    # compute percentage of abs_diff < 0.1
    for thresh in [0.1, 0.15, 0.2]:
        acc_thresh = np.sum(abs_diff < thresh) / len(abs_diff)
        print(f"Percentage of valid predictions (abs_diff < {thresh}): {acc_thresh * 100:.2f}%")
    
    # accuracy per std threshold
    if cfgs.mc_dropout:
        std_thresholds = [0.01 * i for i in range(1, 21)]
        valid_percentages = []
        accuracies = []
        maes = [] # Create a list to store MAE values

        # Ensure all_std_devs is a numpy array for efficient slicing
        all_std_devs = np.array(all_std_devs)

        for std_thresh in std_thresholds:
            std_mask = all_std_devs <= std_thresh
            
            # Handle cases where no data points meet the threshold to avoid errors
            if not np.any(std_mask):
                valid_percentages.append(0)
                accuracies.append(np.nan) # Use NaN for undefined accuracy
                maes.append(np.nan)       # Use NaN for undefined MAE
                continue

            valid_percent = np.sum(std_mask) / len(std_mask) * 100
            accuracy = sklearn.metrics.accuracy_score(all_gt_cls[std_mask], all_pred_cls[std_mask]) * 100
            mae = np.mean(np.abs(all_gt_scores[std_mask] - all_pred_scores[std_mask]))
            print(f"Std Threshold: {std_thresh:.2f}, Valid Percentage: {valid_percent:.2f}%, Accuracy: {accuracy:.2f}%, MAE: {mae:.4f}")
            
            valid_percentages.append(valid_percent)
            accuracies.append(accuracy)
            maes.append(mae) # Store the calculated MAE

        # 2. Plotting with two y-axes
        fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

        # Plot Accuracy and Valid Percentage on the primary y-axis (ax1)
        color1 = 'tab:blue'
        ax1.set_xlabel('Standard Deviation Threshold')
        ax1.set_ylabel('Percentage (%)', color=color1)
        ax1.plot(std_thresholds, accuracies, label='Accuracy (%)', marker='o', color=color1)
        ax1.plot(std_thresholds, valid_percentages, label='Valid Data (%)', marker='x', color='tab:cyan', linestyle='--')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, which='both', linestyle='-.', linewidth=0.5)
        # visualize values
        for i, txt in enumerate(accuracies):
            ax1.annotate(f'{txt:.3}', (std_thresholds[i], accuracies[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=7)

        # Create a second y-axis that shares the same x-axis
        ax2 = ax1.twinx()

        # Plot MAE on the secondary y-axis (ax2)
        color2 = 'tab:red'
        ax2.set_ylabel('MAE', color=color2)
        ax2.plot(std_thresholds, maes, label='MAE', marker='s', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        for i, txt in enumerate(maes):
            ax2.annotate(f'{txt:.3f}', (std_thresholds[i], maes[i]), textcoords="offset points", xytext=(0,-10), ha='center', fontsize=7)

        # Title and Legend
        plt.title('Model Performance vs. Standard Deviation Threshold')

        # For a unified legend, get handles and labels from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')

        # Ensure the plot layout is clean
        fig.tight_layout()

        # Save and close the figure
        plt.savefig(os.path.join(output_dir, 'std_threshold_analysis_{}.png'.format(cfgs.test_split)))
        plt.close()

        print("Plot saved successfully. ✅")

        
        
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset paths
    parser.add_argument('--g1b_root', type=str, help='G1B dataset root directory',
                        default='/home/seung/Workspaces/Datasets/GraspNet-1Billion/grasp_qnet')
    parser.add_argument('--acronym_root', type=str,
                        default='/home/seung/Workspaces/Datasets/ACRONYM/grasp_qnet',
                        help='ACRONYM dataset root directory')
    
    # Model and inference settings
    parser.add_argument('--net', type=str, default='dgcnn', 
                        help='Model to use, [one of pointnet2, dgcnn]')
    parser.add_argument('--ckpt_path', required=True, 
                        help='Model checkpoint path')
    parser.add_argument('--test_split', type=str, default='test_seen',
                        help='Test split to use [test_seen, test_similar, test_novel]')
    parser.add_argument('--batch_size', type=int, default=256, 
                        help='Batch size for inference [default: 32]')
    
    # Output settings
    parser.add_argument('--output_dir', default='inference_results', 
                        help='Directory to save inference results [default: inference_results]')
    parser.add_argument('--mc_dropout', action='store_true',
                        help='Enable MC Dropout for uncertainty estimation')
    
    # Visualization settings
    parser.add_argument('--visualize', action='store_true', 
                        help='Enable visualization of point clouds and predictions')
    parser.add_argument('--max_vis_batches', type=int, default=5,
                        help='Maximum number of batches to visualize [default: 5]')
    parser.add_argument('--max_vis_samples_per_batch', type=int, default=4,
                        help='Maximum number of samples to visualize per batch [default: 4]')
    
    cfgs = parser.parse_args()
    run_inference(cfgs)