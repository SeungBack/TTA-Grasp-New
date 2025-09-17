# Basic Libraries
import os
import numpy as np
import math
import time

# PyTorch Libraries
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Config
from utils.arguments import cfgs

# Local Libraries
from models.economicgrasp_grl import EconomicGraspGRL
from models.loss_economicgrasp import get_loss as get_loss_economicgrasp
# from economic_grasp.dataset.graspnet_grl_dataset import GraspNetGRLDataset, collate_fn
from dataset.graspnet_dataset import GraspNetDataset, collate_fn

# ----------- GLOBAL CONFIG ------------

# Epoch
EPOCH_CNT = 0

# Checkpoint path
CHECKPOINT_PATH = cfgs.checkpoint_path if cfgs.checkpoint_path is not None and cfgs.resume else None

# Logging
if not os.path.exists(cfgs.log_dir):
    os.makedirs(cfgs.log_dir)
LOG_FOUT = open(os.path.join(cfgs.log_dir, 'log_train.txt'), 'a')
LOG_FOUT.write(str(cfgs) + '\n')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


# Create Dataset and Dataloader
TRAIN_SOURCE_DATASET = GraspNetDataset(cfgs.dataset_root, camera=cfgs.camera, split='train',
                                voxel_size=cfgs.voxel_size, num_points=cfgs.num_point, remove_outlier=True,
                                augment=True, load_label=True)
TRAIN_SOURCE_DATALOADER = DataLoader(TRAIN_SOURCE_DATASET, batch_size=4, shuffle=True,
                              num_workers=2, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)

TRAIN_TARGET_DATASET = GraspNetDataset(cfgs.dataset_root, camera=cfgs.grl_camera, split=cfgs.grl_split,
                                voxel_size=cfgs.voxel_size, num_points=cfgs.num_point, remove_outlier=True,
                                augment=True, load_label=False, use_ratio=cfgs.grl_use_ratio)
TRAIN_TARGET_DATALOADER = DataLoader(TRAIN_TARGET_DATASET, batch_size=2, shuffle=True,
                              num_workers=2, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)

# Init the model
net = EconomicGraspGRL(seed_feat_dim=512, is_training=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=cfgs.learning_rate, weight_decay=cfgs.weight_decay)

# Load checkpoint if there is any
start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, start_epoch))


# cosine learning rate decay
def get_current_lr(epoch):
    lr = cfgs.learning_rate
    lr = lr * (math.cos(epoch / cfgs.max_epoch * math.pi) + 1) * 0.5
    return lr


def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# ------TRAINING BEGIN  ------------
def train_one_epoch():
    stat_dict = {}  # collect statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    # set model to training mode
    net.train()
    batch_start_time = time.time()
    data_start_time = time.time()
    num_batches = len(TRAIN_SOURCE_DATALOADER)
    domain_criterion = torch.nn.BCEWithLogitsLoss()

    for batch_idx, source_batch_data in enumerate(TRAIN_SOURCE_DATALOADER):
        target_batch_data = next(iter(TRAIN_TARGET_DATALOADER))
        for key in source_batch_data:
            if 'list' in key:
                for i in range(len(source_batch_data[key])):
                    for j in range(len(source_batch_data[key][i])):
                        source_batch_data[key][i][j] = source_batch_data[key][i][j].to(device)
            else:
                source_batch_data[key] = source_batch_data[key].to(device)
        for key in target_batch_data:
            if 'list' in key:
                for i in range(len(target_batch_data[key])):
                    for j in range(len(target_batch_data[key][i])):
                        target_batch_data[key][i][j] = target_batch_data[key][i][j].to(device)
            else:
                target_batch_data[key] = target_batch_data[key].to(device)
        
        data_end_time = time.time()
        stat_dict['C: Data Time'] = data_end_time - data_start_time

        model_start_time = time.time()
        end_points_source = net(source_batch_data)
        end_points_target = net(target_batch_data)
        model_end_time = time.time()
        stat_dict['C: Model Time'] = model_end_time - model_start_time
        end_points_source['epoch'] = EPOCH_CNT

        loss_start_time = time.time()
        # Compute loss and gradients, update parameters.
        loss, end_points = get_loss_economicgrasp(end_points_source)
        domain_output_source = end_points_source['domain_output']
        
        domain_loss_source = domain_criterion(domain_output_source, torch.zeros_like(domain_output_source)) 
        domain_output_target = end_points_target['domain_output']
        domain_loss_target = domain_criterion(domain_output_target, torch.ones_like(domain_output_target)) 
        loss += domain_loss_source * 0.1
        loss += domain_loss_target * 0.1

        loss.backward()
        if (batch_idx + 1) % 1 == 0:
            optimizer.step()
            optimizer.zero_grad()
        loss_end_time = time.time()
        stat_dict['C: Loss Time'] = loss_end_time - loss_start_time
        end_points['C: Domain Loss Source'] = domain_loss_source
        end_points['C: Domain Loss Target'] = domain_loss_target
        source_domain_acc = ((domain_output_source >= 0) == (torch.zeros_like(domain_output_source) >= 0)).float().mean()
        target_domain_acc = ((domain_output_target >= 0) == (torch.ones_like(domain_output_target) >= 0)).float().mean()
        end_points['C: Domain Acc Source'] = source_domain_acc
        end_points['C: Domain Acc Target'] = target_domain_acc
        # Accumulate statistics and print out
        for key in end_points:
            if 'A' in key or 'B' in key or 'C' in key or 'D' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_interval = 20

        if (batch_idx + 1) % batch_interval == 0:
            remain_batches = (cfgs.max_epoch - EPOCH_CNT) * num_batches - batch_idx - 1
            batch_time = time.time() - batch_start_time
            batch_start_time = time.time()
            stat_dict['C: Remain Time (h)'] = remain_batches * batch_time / 3600
            log_string(f' ---- epoch: {EPOCH_CNT},  batch: {batch_idx + 1} ----')
            for key in sorted(stat_dict.keys()):
                log_string(f'{key:<20}: {round(stat_dict[key] / batch_interval, 4):0<8}')
                stat_dict[key] = 0

        data_start_time = time.time()


def train(start_epoch):
    global EPOCH_CNT
    for epoch in range(start_epoch, cfgs.max_epoch):
        EPOCH_CNT = epoch
        log_string(f'**** EPOCH {epoch:<3} ****')
        log_string('Current learning rate: %f' % (get_current_lr(epoch)))

        np.random.seed()
        train_one_epoch()

        # Save checkpoint
        save_dict = {'epoch': epoch + 1,  # after training one epoch, the start_epoch should be epoch+1
                     'optimizer_state_dict': optimizer.state_dict(),
                     }
        try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = net.module.state_dict()
        except:
            save_dict['model_state_dict'] = net.state_dict()
        torch.save(save_dict, os.path.join(cfgs.log_dir, cfgs.model + '_epoch' + str(epoch + 1).zfill(2) + '.tar'))


if __name__ == '__main__':
    train(start_epoch)
