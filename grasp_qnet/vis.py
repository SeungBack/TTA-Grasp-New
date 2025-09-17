import os
import sys
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import the dataset class (assuming the same import from original code)
from dataset.data_loader import GraspEvalDataset

# Import the models
from models.pointnet_v2 import *
from models.dgcnn import *
import open3d as o3d

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
    
    # Create output directories
    output_dir = cfgs.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create visualization directory
    
    # Set up logging
    log_dir = output_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_fout = open(os.path.join(log_dir, 'log_inference.txt'), 'w')
    log_fout.write(str(cfgs) + '\n')
    log_fout.close()

    # Load the test dataset
    log_string(f'Loading {cfgs.test_split} dataset...', log_dir)
    test_dataset = GraspEvalDataset(cfgs.g1b_root, cfgs.acronym_root, split=cfgs.test_split, return_score=True)
    test_dataloader = DataLoader(test_dataset, batch_size=cfgs.batch_size, shuffle=True)
    log_string(f'Test dataset size: {len(test_dataset)}', log_dir)

    # Initialize model
    log_string(f'Initializing {cfgs.net} model...', log_dir)
    if cfgs.net == 'pointnet2':
        net = PointNet2GraspQNet()
    elif cfgs.net == 'dgcnn':
        net = DGCNNGraspQNet()
    elif cfgs.net == 'dgcnn_new':
        net = ImprovedDGCNNGraspQNet()
    else:
        raise ValueError('Network not supported')

    # Load model to GPU(s)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        log_string(f'Using {torch.cuda.device_count()} GPUs for inference!', log_dir)
        net = torch.nn.DataParallel(net)
    net = net.to(device)

    # Load checkpoint
    if cfgs.ckpt_path:
        log_string(f'Loading checkpoint from {cfgs.ckpt_path}...', log_dir)
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
            
            log_string('Checkpoint loaded successfully!', log_dir)
        except Exception as e:
            log_string(f'Error loading model state: {str(e)}', log_dir)
            raise
    else:
        log_string('No checkpoint provided. Using randomly initialized model.', log_dir)

    # Print model parameter count
    num_params = sum(p.numel() for p in net.parameters())
    log_string(f'Number of parameters: {num_params}', log_dir)

    # Run inference
    log_string('Starting inference...', log_dir)
    net.eval()  # Set the model to evaluation mode
    
    # Initialize statistics
    total_samples = 0
    correct_predictions = 0
    all_predictions = []
    all_ground_truths = []
    idx = 0
    with torch.no_grad():  # Disable gradient computation
        for batch_idx, (obj_cloud, gripper_cloud, category) in enumerate(tqdm(test_dataloader)):
            # Move data to device
            obj_cloud = obj_cloud.to(device)
            gripper_cloud = gripper_cloud.to(device)
            gt_score = category.to(device)
            
            # Forward pass
            pred_scores = net(obj_cloud, gripper_cloud)
            
            
            # Visualize samples if requested
            for i in range(obj_cloud.size(0)):
                # Get sample data
                obj_points = obj_cloud[i].cpu().numpy()
                gripper_points = gripper_cloud[i].cpu().numpy()
                pred = pred_scores[i].item()  # Probability of success
                gt = gt_score[i].item()
                all_predictions.append(pred)
                all_ground_truths.append(gt)
                
                obj_o3d = o3d.geometry.PointCloud()
                obj_o3d.points = o3d.utility.Vector3dVector(obj_points)
                gripper_o3d = o3d.geometry.PointCloud()
                gripper_o3d.points = o3d.utility.Vector3dVector(gripper_points)
                print('gt: {}, pred: {}'.format(gt, pred))
                o3d.visualization.draw_geometries([obj_o3d, gripper_o3d])
            idx += 1
            if idx > 10:
                break
    
    # Visualize the comparison of predictions and ground truths with bar plots

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Generate sample data (replace this with your actual data)
    np.random.seed(42)  # For reproducibility
    n_samples = len(all_ground_truths)  # Number of samples
    all_ground_truths = np.array(all_ground_truths)
    all_predictions = np.array(all_predictions)
    gt_score = all_ground_truths
    # Create predictions with some noise to simulate imperfect predictions
    pred_score = all_predictions

    # Calculate error metrics
    mae = mean_absolute_error(gt_score, pred_score)
    rmse = np.sqrt(mean_squared_error(gt_score, pred_score))
    r2 = r2_score(gt_score, pred_score)

    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Scatter plot with identity line
    ax1.scatter(gt_score, pred_score, alpha=0.6, c='blue', label='Predictions')
    # Add identity line (perfect prediction line)
    identity_line = np.linspace(0, 1, 100)
    ax1.plot(identity_line, identity_line, 'r--', label='Perfect Prediction')

    # Add error stats to the plot
    ax1.text(0.05, 0.95, f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}', 
            transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    ax1.set_xlabel('Ground Truth Score')
    ax1.set_ylabel('Prediction Score')
    ax1.set_title('Prediction vs Ground Truth (Scatter)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # 2. Line plot comparing sorted values
    # Sort both arrays for better visualization in line plot
    sorted_indices = np.argsort(gt_score)
    gt_sorted = gt_score[sorted_indices]
    pred_sorted = pred_score[sorted_indices]

    ax2.plot(range(n_samples), gt_sorted, 'g-', label='Ground Truth', linewidth=2)
    ax2.plot(range(n_samples), pred_sorted, 'b-', label='Predictions', linewidth=2, alpha=0.7)
    ax2.fill_between(range(n_samples), gt_sorted, pred_sorted, color='red', alpha=0.2, label='Difference')

    ax2.set_xlabel('Sample Index (sorted by ground truth)')
    ax2.set_ylabel('Score')
    ax2.set_title('Ground Truth vs Prediction (Sorted)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0, n_samples-1)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

    # Additional visualization: Error distribution histogram
    plt.figure(figsize=(8, 5))
    errors = pred_score - gt_score
    plt.hist(errors, bins=20, alpha=0.7, color='blue')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=1)
    plt.xlabel('Prediction Error (pred - gt)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    

            


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
    parser.add_argument('--test_split', type=str, default='test_novel',
                        help='Test split to use [test_seen, test_similar, test_novel]')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for inference [default: 32]')
    
    # Output settings
    parser.add_argument('--output_dir', default='inference_results', 
                        help='Directory to save inference results [default: inference_results]')
    
    # Visualization settings
    
    cfgs = parser.parse_args()
    run_inference(cfgs)