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




BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))
# sys.path.append('/SSD1/Workspace/grasping_baselines/contact-graspnet/tools')
import builder

from grasp_toolkit.simulation.src.vgn.experiments import clutter_removal
from grasp_toolkit.simulation.src.vgn.utils.misc import set_random_seed
from grasp_toolkit.simulation.src.vgn.grasp import Grasp as G
from grasp_toolkit.simulation.src.vgn.utils.transform import Transform


# from utils.config import *
# import utils.utils as custom_utils
# from collision_detector import ModelFreeCollisionDetector
from graspnetAPI import GraspGroup, Grasp
from scipy.spatial.transform import Rotation as R
from omegaconf import OmegaConf

from models.economicgrasp import EconomicGrasp, pred_decode, load_economicgrasp


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grasp_toolkit.tta import get_tta_method
from grasp_toolkit.utils.collision_detector import ModelFreeCollisionDetector


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



def main(args):
    
    cfg = OmegaConf.load(args.cfg)
    if 'dump_dir' not in cfg.keys() or cfg.dump_dir == '':
        cfg.dump_dir = args.cfg.replace('configs', 'logs').replace('.yaml', '')
    print('dump_dir: ', cfg.dump_dir)
    base_cfg = OmegaConf.load(cfg.base_cfg)
    cfg = OmegaConf.merge(base_cfg, cfg)

    if args.type in ['giga', 'giga_aff']:
        grasp_planner = EconomicGrasp(cfg)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default='')
    parser.add_argument("--type", type=str, default='giga')
    parser.add_argument("--logdir", type=Path, default="data/experiments")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--scene",
                        type=str,
                        choices=["pile", "packed"],
                        default="pile")
    parser.add_argument("--object-set", type=str, default="blocks")
    parser.add_argument("--num-objects", type=int, default=5)
    parser.add_argument("--num-view", type=int, default=1)
    parser.add_argument("--num-rounds", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs='+', default=[0, 1, 2, 3, 4])
    # parser.add_argument("--seeds", type=int, nargs='+', default=[0])
    parser.add_argument("--sim-gui", action="store_true")
    # parser.add_argument("--grad-refine", action="store_true")
    parser.add_argument("--qual-th", type=float, default=0.9)
    parser.add_argument("--eval-geo",
                        action="store_true",
                        help='whether evaluate geometry prediction')
    parser.add_argument(
        "--best",
        action="store_true",
        help="Whether to use best valid grasp (or random valid grasp)")
    parser.add_argument("--result-path", type=str)
    parser.add_argument(
        "--force",
        action="store_true",
        help=
        "When all grasps are under threshold, force the detector to select the best grasp"
    )
    parser.add_argument(
        "--add-noise",
        type=str,
        default='',
        help="Whether add noise to depth observation, trans | dex | norm | ''")
    parser.add_argument("--sideview",
                        action="store_true",
                        help="Whether to look from one side")
    parser.add_argument("--silence",
                        action="store_true",
                        help="Whether to disable tqdm bar")
    parser.add_argument("--vis",
                        action="store_true",
                        help="visualize and save affordance")
    
    parser.add_argument(
        '--cfg',
        type=str,
        default='../../economic_grasp/configs/graspnet1b/notta/realsense_similar.yaml'
    )
    parser.add_argument(
        '--collision_thresh',
        type=float,
        default=0.01
    )


    args = parser.parse_args()
    main(args)
