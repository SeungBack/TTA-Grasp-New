from .base import TTA_Base
from .norm import Norm
from .cotta import CoTTA
from .tta_grasp import TTA_Grasp_GraspNetBaseline, TTA_Grasp_EconomicGrasp
from .tent import TENT



def get_tta_method(cfg, model):
    if cfg.tta.method == 'notta':
        return TTA_Base(cfg, model)
    elif cfg.tta.method in ['bn-1', 'bn-adapt', 'bn-ema']:
        return Norm(cfg, model)
    elif cfg.tta.method == 'cotta':
        return CoTTA(cfg, model)
    elif cfg.tta.method == 'tta-grasp':
        if cfg.model.name == 'graspnet_baseline':
            return TTA_Grasp_GraspNetBaseline(cfg, model)
        elif cfg.model.name == 'economic_grasp':
            return TTA_Grasp_EconomicGrasp(cfg, model)
        else:
            raise ValueError(f'Invalid model name for TTA: {cfg.model.name}')
    
    elif cfg.tta.method == 'tent':
        return TENT(cfg, model)
    else:
        raise ValueError(f'Invalid TTA method: {cfg.tta.method}')

