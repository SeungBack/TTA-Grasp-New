import argparse
import numpy as np
import json
from pathlib import Path

import argparse
import os
import numpy as np
np.float = np.float64  # temp fix for following import
import torch
import open3d as o3d
import sys
from omegaconf import OmegaConf




BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))
import builder

from grasp_sim.experiments import clutter_removal
from grasp_sim.utils.misc import set_random_seed
from grasp_sim.grasp import Grasp as G
from grasp_sim.utils.transform import Transform

from utils.config import *
import utils.utils as custom_utils
from collision_detector import ModelFreeCollisionDetector
from graspnetAPI import GraspGroup, Grasp
from scipy.spatial.transform import Rotation as R

from model_wrapper import EconomicGraspPlanner


def main(args):


    if args.model == 'economic_grasp':
        grasp_planner = AnyGraspNet(args)
    
    else:
        raise NotImplementedError(f'model type {args.type} not implemented!')

    gsr = []
    dr = []
    for seed in args.seeds:
        set_random_seed(seed)
        success_rate, declutter_rate = clutter_removal.run(
            grasp_plan_fn=grasp_planner,
            logdir=args.logdir,
            description=args.description,
            scene=args.scene,
            object_set=args.object_set,
            num_objects=args.num_objects,
            n=args.num_view,
            num_rounds=args.num_rounds,
            seed=seed,
            sim_gui=args.sim_gui,
            result_path=None,
            add_noise=args.add_noise,
            sideview=args.sideview,
            silence=args.silence,
            visualize=args.vis,)
        gsr.append(success_rate)
        dr.append(declutter_rate)
    results = {
        'gsr': {
            'mean': np.mean(gsr),
            'std': np.std(gsr),
            'val': gsr
        },
        'dr': {
            'mean': np.mean(dr),
            'std': np.std(dr),
            'val': dr
        }
    }
    print('Average results:')
    print(f'Grasp sucess rate: {np.mean(gsr):.2f} ± {np.std(gsr):.2f} %')
    print(f'Declutter rate: {np.mean(dr):.2f} ± {np.std(dr):.2f} %')
    with open(args.result_path, 'w') as f:
        json.dump(results, f, indent=2)



if __name__ == "__main__":
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--tta-cfgs', type=str, default=None, help='Path to tta cfgs file')
    parser.add_argument('--sim-cfgs', type=str, default=None, help='Path to sim cfgs file')
    parser.add_argument("--sim-gui", action="store_true")
    args = parser.parse_args()
    
    if args.debug:
        print("Debug mode active, wandb logging will be disabled")
    
    # Load configuration
    cfgs = OmegaConf.load(args.cfgs)
    if 'dump_dir' not in cfgs.keys() or cfgs.dump_dir == '':
        cfgs.dump_dir = args.cfgs.replace('configs', 'logs').replace('.yaml', '')
    print('dump_dir: ', cfgs.dump_dir)
    base_cfgs = OmegaConf.load(cfgs.base_cfgs)
    cfgs = OmegaConf.merge(base_cfgs, cfgs)


    # parser = argparse.ArgumentParser()
    # parser.add_argument("--num-objects", type=int, default=5)
    # parser.add_argument("--num-view", type=int, default=1)
    # parser.add_argument("--num-rounds", type=int, default=100)
    # parser.add_argument("--seeds", type=int, nargs='+', default=[0, 1, 2, 3, 4])
    # parser.add_argument("--qual-th", type=float, default=0.9)

    # parser.add_argument(
    #     "--best",
    #     action="store_true",
    #     help="Whether to use best valid grasp (or random valid grasp)")
    # parser.add_argument("--result-path", type=str)
    # parser.add_argument(
    #     "--force",
    #     action="store_true",
    #     help=
    #     "When all grasps are under threshold, force the detector to select the best grasp"
    # )
    # parser.add_argument(
    #     "--add-noise",
    #     type=str,
    #     default='',
    #     help="Whether add noise to depth observation, trans | dex | norm | ''")
    # parser.add_argument("--sideview",
    #                     action="store_true",
    #                     help="Whether to look from one side")
    # parser.add_argument("--silence",
    #                     action="store_true",
    #                     help="Whether to disable tqdm bar")
    # parser.add_argument("--vis",
    #                     action="store_true",
    #                     help="visualize and save affordance")
    



    main(cfgs)