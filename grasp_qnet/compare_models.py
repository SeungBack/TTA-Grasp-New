import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import time

from models.pointnet_v2 import PointNet2GraspQNet
from models.dgcnn import DGCNNGraspQNet, ImprovedDGCNNGraspQNet
from models.pct import PCTGraspQNet
from models.pointtransformer import PointTransformerGraspQNet
from models.pointmlp import PointMLPGraspQNet
from models.edgegrasp import EdgeGraspQNet

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, '../ext'))
from pointnet2.pytorch_utils import BNMomentumScheduler

import time

if __name__ == '__main__':
    
    nets = ['pointnet2', 'dgcnn', 'pct', 'pointtransformer']
    nets = ['edgegrasp']
    for net in nets:
        print(f'----Running {net}...-------')
        if net == 'pointnet2':
            net = PointNet2GraspQNet()
        elif net == 'dgcnn':
            net = DGCNNGraspQNet()
        elif net == 'dgcnn_new':
            net = ImprovedDGCNNGraspQNet()
        elif net == 'pct':
            net = PCTGraspQNet()
        elif net == 'pointtransformer':
            net = PointTransformerGraspQNet()
        elif net == 'pointmlp':
            net = PointMLPGraspQNet()
        elif net == 'edgegrasp':
            net = EdgeGraspQNet()
        else:
            raise ValueError('Network not supported')
        
        net = net.cuda()
        # print number of parameters
        num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        # print with M unit
        print(f'Number of parameters: {num_params/1e6:.2f}M')
        
        
        times = []
        input = torch.randn(256, 1024, 3).cuda()
        gripper = torch.randn(256, 128, 3).cuda()
        for i in range(1001):
            start_time = time.time()
            output = net(input, gripper)
            if i == 0:
                continue
            times.append(time.time() - start_time)
            
        print(f'Average time: {np.mean(times)}')
        print(f'Max time: {np.max(times)}')
        print(f'Min time: {np.min(times)}')
        print(f'Time std: {np.std(times)}')
        print(f'FPS: {1/np.mean(times)}')
        
        
    