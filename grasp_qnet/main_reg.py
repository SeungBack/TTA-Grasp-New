import os
import random
import sys
import numpy as np
from datetime import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.data_loader import GraspEvalDataset, create_balanced_dataloader
from tqdm import tqdm
import random
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

import torch.nn.functional as F
from data_utils import AdamW
from torch.optim import Adam, SGD
import sklearn

from models.pointnet_v2 import *
from models.dgcnn import *
from models.edgegrasp import *
from models.pointmlp import *

from cbloss.loss import FocalLoss, ClassBalancedLoss
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, '..'))


def init_stats(split):
    stat =  {
        f'{split}_loss': 0,
        f'{split}_mae_acc': 0,
    }
    for th in [0.3, 0.5]:
        stat[f'{split}_acc_{th}'] = 0
        stat[f'{split}_bal_acc_{th}'] = 0
        stat[f'{split}_acc_1_{th}'] = 0
        stat[f'{split}_acc_0_{th}'] = 0
    return stat
            

def log_string(out_str, log_dir):
    with open(os.path.join(log_dir, 'log_train.txt'), 'a') as log_fout:
        log_fout.write(out_str + '\n')
        log_fout.flush()
    print(out_str)


def train(cfgs):
    # TensorBoard Visualizers
    log_dir = cfgs.log_dir
    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_fout = open(os.path.join(log_dir, 'log_train.txt'), 'a')
    log_fout.write(str(cfgs) + '\n')

    train_dataset = GraspEvalDataset(cfgs.g1b_root, cfgs.acronym_root, split='train', use_normal=cfgs.use_normal, return_score=True)
    train_dataloader = DataLoader(train_dataset, batch_size=cfgs.batch_size, shuffle=True, num_workers=cfgs.num_workers)
    # train_dataloader = create_balanced_dataloader(train_dataset, cfgs.batch_size, cfgs.real_ratio, cfgs.num_workers)
    print('train dataset size:', len(train_dataset))

    test_seen_dataset = GraspEvalDataset(cfgs.g1b_root, cfgs.acronym_root, split='test_seen', use_normal=cfgs.use_normal, return_score=True)
    test_seen_dataloader = DataLoader(test_seen_dataset, batch_size=cfgs.batch_size, shuffle=False, num_workers=1)
    
    test_similar_dataset = GraspEvalDataset(cfgs.g1b_root, cfgs.acronym_root, split='test_similar', use_normal=cfgs.use_normal, return_score=True)
    test_similar_dataloader = DataLoader(test_similar_dataset, batch_size=cfgs.batch_size, shuffle=False, num_workers=1)
    
    test_novel_dataset = GraspEvalDataset(cfgs.g1b_root, cfgs.acronym_root, split='test_novel', use_normal=cfgs.use_normal, return_score=True)
    test_novel_dataloader = DataLoader(test_novel_dataset, batch_size=cfgs.batch_size, shuffle=False, num_workers=1)
    
    # Store test dataloaders in a list for easier iteration
    test_dataloaders = [test_seen_dataloader, test_similar_dataloader, test_novel_dataloader]
    test_splits = ['test_seen', 'test_similar', 'test_novel']
    
    if cfgs.net == 'pointnet2':
        net = PointNet2GraspQNet()
    elif cfgs.net == 'dgcnn':
        net = DGCNNGraspQNet(use_normal=cfgs.use_normal, dropout=cfgs.dropout, k=cfgs.k)
    elif cfgs.net == 'edgegrasp':
        net = EdgeGraspQNet(use_normal=cfgs.use_normal)
    elif cfgs.net == 'pointmlp':
        net = PointMLPGraspQNet()
    else:
        raise ValueError('Network not supported')

    # support multi gpu
    device_ids = [i for i in range(torch.cuda.device_count())]
    if len(device_ids) > 1:
        print('Using', device_ids, 'GPUs!')
        net = torch.nn.DataParallel(net, device_ids=device_ids)
    net = net.cuda()
    print('Using', device_ids, 'GPUs!')

    net.train()
    optimizer = AdamW(net.parameters(), lr=cfgs.learning_rate, weight_decay=cfgs.weight_decay)
    
    # Setup for per-iteration warmup + step LR schedule
    warmup_iters = cfgs.warmup_iters  # Direct control over iterations, not epochs
    warmup_factor = cfgs.warmup_factor
    
    # Create custom LR scheduler that handles both warmup and step decay
    def custom_lr_scheduler(optimizer, warmup_iters, warmup_factor, step_size, gamma, min_lr):
        # Initial LR
        init_lr = optimizer.param_groups[0]['lr']
        
        def lr_lambda(iteration):
            # Handle case where warmup_iters is 0 (no warmup)
            if warmup_iters <= 0:
                # Just apply step decay
                num_decays = iteration // step_size
                decay_factor = gamma ** num_decays
                decay_factor = max(decay_factor, min_lr / init_lr)
                return decay_factor
                
            # Warmup phase: linearly increase from warmup_factor*base_lr to base_lr
            if iteration < warmup_iters:
                alpha = float(iteration) / float(warmup_iters)
                return warmup_factor * (1 - alpha) + alpha
            
            # Step decay phase: decay by gamma every step_size iterations
            # Calculate how many step decays have occurred
            num_decays = (iteration - warmup_iters) // step_size
            decay_factor = gamma ** num_decays
            
            # Ensure we don't go below minimum learning rate
            decay_factor = max(decay_factor, min_lr / init_lr)
            
            return decay_factor
            
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Create the combined scheduler
    scheduler = custom_lr_scheduler(
        optimizer, 
        warmup_iters=warmup_iters,
        warmup_factor=warmup_factor,
        step_size=cfgs.step_size_iters,
        gamma=cfgs.step_gamma,
        min_lr=cfgs.min_learning_rate
    )
    
    if warmup_iters > 0:
        log_string(f"Using per-iteration warmup for {warmup_iters} iterations starting at " 
                   f"{cfgs.learning_rate * warmup_factor:.6f} and increasing to {cfgs.learning_rate:.6f}", log_dir)
    else:
        log_string(f"No warmup phase - starting directly at learning rate {cfgs.learning_rate:.6f}", log_dir)
    log_string(f"LR will decay by {cfgs.step_gamma} every {cfgs.step_size_iters} iterations", log_dir)

    # Calculate evaluation frequency
    total_train_batches = len(train_dataloader)
    eval_frequency = cfgs.eval_frequency if hasattr(cfgs, 'eval_frequency') else total_train_batches // cfgs.evals_per_epoch
    log_string(f'Total batches per epoch: {total_train_batches}', log_dir)
    log_string(f'Will evaluate model every {eval_frequency} batches', log_dir)

    # Initialize tracking variables
    start_epoch = 0
    global_iter = 0
    best_accuracy = 0
    best_model_path = None
    
    # Load checkpoint if resuming
    if cfgs.resume:
        checkpoint = torch.load(cfgs.ckpt_path)
        try:
            net.load_state_dict(checkpoint['model_state_dict'])
        except:
            # Handle DataParallel case
            try:
                # If current model is not DataParallel but saved model was
                if 'module.' in list(checkpoint['model_state_dict'].keys())[0]:
                    new_state_dict = {}
                    for k, v in checkpoint['model_state_dict'].items():
                        name = k[7:]  # remove 'module.' prefix
                        new_state_dict[name] = v
                    net.load_state_dict(new_state_dict)
                else:
                    # If there's some other issue, try the module approach
                    net.module.load_state_dict(checkpoint['model_state_dict'])
            except Exception as e:
                log_string(f"Error loading model state: {str(e)}", log_dir)
                raise
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                log_string("Loaded scheduler state from checkpoint", log_dir)
            except Exception as e:
                log_string(f"Could not load scheduler state: {str(e)}", log_dir)
                log_string("Will rebuild scheduler state from iteration count", log_dir)
        
        # Load training state
        start_epoch = checkpoint['epoch']
        if 'iter' in checkpoint:
            global_iter = checkpoint['iter']
            log_string(f"Resuming from iteration {global_iter}", log_dir)
            
            # If we couldn't load scheduler state directly, rebuild it
            if 'scheduler_state_dict' not in checkpoint or checkpoint['scheduler_state_dict'] is None:
                # When resuming, rebuild scheduler state by stepping to current iteration
                for _ in range(global_iter):
                    scheduler.step()
                log_string(f"Rebuilt scheduler state to iteration {global_iter}", log_dir)
        
        if 'best_accuracy' in checkpoint:
            best_accuracy = checkpoint['best_accuracy']
            log_string(f"Loaded previous best accuracy: {best_accuracy:.4f}", log_dir)
    elif cfgs.pretrained:
         # load nn.DataParallel to nn.DataParallel
        weight = torch.load(cfgs.ckpt_path)
        new_weight = {}
        for k, v in weight.items():
            if 'module.' in k:
                k = k[7:]
            if 'score_head' in k:
                continue
            new_weight[k] = v
        net.load_state_dict(new_weight, strict=False)
        
    # print number of parameters for each part
    for name, param in net.named_parameters():
        if param.requires_grad:
            log_string(f'Parameter {name} has {param.numel()/1e6:.2f}M parameters', log_dir)
    
    log_string('Number of parameters: {}'.format(sum([p.numel() for p in net.parameters()])), log_dir)

    # test FPS of model
    net.eval()
    with torch.no_grad():
        dummy_obj = torch.randn(1, 2048, 3).cuda()
        dummy_gripper = torch.randn(1, 128, 3).cuda()
        dummy_score = torch.randn(1, 1).cuda()
        
        start_time = datetime.now()
        for _ in range(100):
            _ = net(dummy_obj, dummy_gripper)
        end_time = datetime.now()
        
        elapsed_time = (end_time - start_time).total_seconds()
        fps = 100 / elapsed_time
        log_string(f"Model FPS: {fps:.2f}", log_dir)

    # Main training loop
    for epoch in range(start_epoch, cfgs.max_epoch):
        log_string('**** EPOCH %03d ****' % (epoch), log_dir)
        log_string('Current learning rate: %f'%(optimizer.param_groups[0]['lr']), log_dir)
        log_string(str(datetime.now()), log_dir)
        
        # Reset seeds for reproducibility
        np.random.seed(777)
        random.seed(777)
        torch.manual_seed(777)
        
        # Training phase
        net.train()
        train_stats = init_stats('train')
        batch_count = 0
        # criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for binary classification
        # criterion = FocalLoss(num_classes=2, gamma=2.0, alpha=0.25, reduction='mean')
        # for balanced loss, compute number of samples per class by iterating through the dataset
        # num_cls_0 = 0
        # num_cls_1 = 0
        # for obj_cloud, gripper_cloud, gt_label in train_dataloader:
        #     gt_label = gt_label.numpy()
        #     num_cls_0 += np.sum(gt_label == 0)
        #     num_cls_1 += np.sum(gt_label == 1)
        # print(f'Number of samples in class 0: {num_cls_0}, class 1: {num_cls_1}')
        criterion = nn.SmoothL1Loss(beta=0.1)  # Use SmoothL1Loss for regression
        
        
        for batch_idx, (obj_cloud, gripper_cloud, gt_score) in enumerate(tqdm(train_dataloader)):
            
            global_iter += 1
            # Zero gradients
            optimizer.zero_grad()
            
            pred = net(obj_cloud.to('cuda'), gripper_cloud.to('cuda')).squeeze(-1)
            gt_score = gt_score.to('cuda').float()
            gt_score = torch.round(gt_score, decimals=2)  # Round to 2 decimal places
            
            loss = criterion(pred, gt_score)
            train_stats['train_loss'] += loss.item()
            
            loss.backward()
            optimizer.step()
            
            for th in [0.3, 0.5]:
                pred_label = (pred >= th).long().cpu().numpy()
                gt_label = (gt_score >= th).long().cpu().numpy()
                acc = sklearn.metrics.accuracy_score(gt_label, pred_label)
                bal_acc = sklearn.metrics.balanced_accuracy_score(gt_label, pred_label)
                cls_1_acc = sklearn.metrics.precision_score(gt_label, pred_label, pos_label=1, zero_division=0.0)
                cls_0_acc = sklearn.metrics.precision_score(gt_label, pred_label, pos_label=0, zero_division=0.0)
                train_stats[f'train_acc_{th}'] += acc
                train_stats[f'train_bal_acc_{th}'] += bal_acc
                train_stats[f'train_acc_1_{th}'] += cls_1_acc
                train_stats[f'train_acc_0_{th}'] += cls_0_acc
                
            # compute MAE accuracy
            mae_acc = np.mean(np.abs(pred.detach().cpu().numpy() - gt_score.cpu().numpy()))
            train_stats['train_mae_acc'] += mae_acc
            
            
            batch_count += 1
            
            # Log training stats periodically
            if (batch_idx + 1) % 100 == 0:
                print(' ---- batch: %03d ----' % (batch_idx + 1))
                for key in sorted(train_stats.keys()):
                    avg_value = train_stats[key] / batch_count
                    train_writer.add_scalar(f'train/{key}', avg_value, global_iter)
                    log_string('%s: %f' % (key, avg_value), log_dir)
                # Reset stats
                train_stats = init_stats('train')
                batch_count = 0
            
            # Evaluate model periodically
            if (global_iter + 1) % 1000 == 0 or batch_idx + 1 == total_train_batches:
                log_string(f'Evaluating at epoch {epoch}, batch {batch_idx+1}/{total_train_batches}', log_dir)
                
                # Switch to evaluation mode is handled in evaluate function
                
                # Evaluate on all test datasets
                for test_loader, split_name in zip(test_dataloaders, test_splits):
                    evaluate(cfgs, net, test_loader, train_writer, split_name, global_iter, criterion)
                
                # Save checkpoint
                checkpoint_path = os.path.join(cfgs.log_dir, f'checkpoint_epoch{epoch}_global{global_iter}.tar')
                save_checkpoint(net, optimizer, epoch, global_iter, loss, best_accuracy, scheduler, checkpoint_path)
                
                # Switch back to training mode after evaluation
                net.train()
            
            # Update learning rate scheduler after every batch
            scheduler.step()
            
            # Log learning rate periodically
            if (global_iter + 1) % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                if global_iter < warmup_iters and warmup_iters > 0:
                    log_string(f"Warmup iter {global_iter+1}/{warmup_iters}, lr: {current_lr:.6f}", log_dir)
                else:
                    log_string(f"Training iter {global_iter+1}, lr: {current_lr:.6f}", log_dir)
        
        # Save epoch checkpoint
        epoch_checkpoint_path = os.path.join(cfgs.log_dir, f'checkpoint_epoch{epoch}.tar')
        save_checkpoint(net, optimizer, epoch + 1, global_iter, loss, best_accuracy, scheduler, epoch_checkpoint_path)
        
        # Explicitly clean up GPU memory
        torch.cuda.empty_cache()

    # Final logging
    log_string(f'Training completed. Best accuracy: {best_accuracy:.4f}', log_dir)
    if best_model_path:
        log_string(f'Best model saved at: {best_model_path}', log_dir)


def evaluate(cfgs, net, dataloader, writer, split, iter_num, criterion):
    """Evaluate the model on a dataset"""
    
    stat_dict = init_stats(split)
    
    # For per-batch class metrics to calculate balanced accuracy properly
    n_batches = 0
    
    # Set model to evaluation mode
    net.eval()
    with torch.no_grad():  # Important: disable gradient computation for evaluation
        for obj_cloud, gripper_cloud, gt_score in dataloader:

            pred = net(obj_cloud.to('cuda'), gripper_cloud.to('cuda')).squeeze(1)
            gt_score = gt_score.to('cuda').float()
            gt_score = torch.round(gt_score, decimals=2)  # Round to 2 decimal places
            loss = criterion(pred, gt_score)
            stat_dict[f'{split}_loss'] += loss.item()
            
            for th in [0.3, 0.5]:
                pred_label = (pred >= th).long().cpu().numpy()
                gt_label = (gt_score >= th).long().cpu().numpy()
                acc = sklearn.metrics.accuracy_score(gt_label, pred_label)
                bal_acc = sklearn.metrics.balanced_accuracy_score(gt_label, pred_label)
                cls_1_acc = sklearn.metrics.precision_score(gt_label, pred_label, pos_label=1, zero_division=0.0)
                cls_0_acc = sklearn.metrics.precision_score(gt_label, pred_label, pos_label=0, zero_division=0.0)
                
                stat_dict[f'{split}_acc_{th}'] += acc
                stat_dict[f'{split}_bal_acc_{th}'] += bal_acc
                stat_dict[f'{split}_acc_1_{th}'] += cls_1_acc
                stat_dict[f'{split}_acc_0_{th}'] += cls_0_acc
            
            mae_acc = np.mean(np.abs(pred.cpu().numpy() - gt_score.cpu().numpy()))
            stat_dict[f'{split}_mae_acc'] += mae_acc
           
            n_batches += 1
    
    # log results
    log_string(' ---- %s ----' % (split), cfgs.log_dir)
    for key in sorted(stat_dict.keys()):
        avg_value = stat_dict[key] / n_batches
        writer.add_scalar(f'{split}/{key}', avg_value, iter_num)
        log_string('%s: %f' % (key, avg_value), cfgs.log_dir)
    
    return 


def save_checkpoint(net, optimizer, epoch, iter_num, loss, best_accuracy, scheduler, save_path):
    """Save model checkpoint"""
    save_dict = {
        'epoch': epoch,
        'iter': iter_num,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
        'loss': loss,
        'best_accuracy': best_accuracy
    }
    
    try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
        save_dict['model_state_dict'] = net.module.state_dict()
    except:
        save_dict['model_state_dict'] = net.state_dict()
    
    torch.save(save_dict, save_path)
    print(f"Checkpoint saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--g1b_root', type=str, help='G1B dataset root directory', 
                        default='/home/seung/Workspaces/Datasets/GraspNet-1Billion/grasp_qnet')
    parser.add_argument('--acronym_root', type=str,
                        default= '/home/seung/Workspaces/Datasets/ACRONYM/grasp_qnet',
                        help='ACRONYM dataset root directory')
    parser.add_argument('--net', type=str, default='pointnet2', help='Model to use, [one of pointnet2, dgcnn, pointmlp]')
    parser.add_argument('--ckpt_path', default=None, help='Model checkpoint path [default: None]')
    parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
    parser.add_argument('--max_epoch', type=int, default=200, help='Epoch to run [default: 18]')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default:24]')
    parser.add_argument('--num_workers', type=int, default=2, help='workers num during training [default: 2]')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--min_learning_rate', type=float, default=1e-5, help='Minimum learning rate [default: 1e-5]')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Optimization L2 weight decay [default: 0]')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--pretrained', action='store_true', help='Load pretrained model')
    # New arguments for frequent evaluation
    parser.add_argument('--evals_per_epoch', type=int, default=1, 
                        help='Number of evaluations to perform per epoch [default: 4]')
    parser.add_argument('--save_best_only', action='store_true', 
                        help='Only save checkpoints when model improves (except epoch end)')
    parser.add_argument('--warmup_iters', type=int, default=5000, 
                        help='Number of iterations for learning rate warmup [default: 5000]')
    parser.add_argument('--warmup_factor', type=float, default=0.1, 
                        help='Factor to multiply learning rate by at the start of warmup [default: 0.1]')
    parser.add_argument('--step_size_iters', type=int, default=10000, 
                        help='Step LR period in iterations [default: 1000]')
    parser.add_argument('--step_gamma', type=float, default=0.7, 
                        help='Step LR decay factor [default: 0.7]')
    # Advanced optimization options
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='Clip gradients to this maximum value (0 for no clipping)')
    parser.add_argument('--use_normal', action='store_true', 
                        help='Use normal information')
    parser.add_argument('--real_ratio', type=float, default=0.5,
                        help='Real data ratio in the batch [default: 0.5]')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate for the model [default: 0.5]')
    parser.add_argument('--k', type=int, default=20,
                        help='Number of nearest neighbors for DGCNN [default: 20]')
    cfgs = parser.parse_args()
    train(cfgs)