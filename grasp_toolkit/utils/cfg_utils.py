"""Configuration utilities for GraspNet testing."""

import os
import yaml
from easydict import EasyDict
import datetime

def load_cfg(cfg_path):
    """
    Load cfguration from YAML file.
    
    Args:
        cfg_path (str): Path to the cfguration file.
        
    Returns:
        EasyDict: Configuration dictionary.
    """
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Convert to EasyDict for easier access
    cfg = EasyDict(cfg)
    
    return cfg


def print_cfg(cfg):
    """
    Print cfguration.
    
    Args:
        cfg (EasyDict): Configuration dictionary.
    """
    print('------------------- cfg -------------------')
    print("Configuration:")
    for section, params in cfg.items():
        if isinstance(params, dict):
            print(f"\n[{section}]")
            for key, value in params.items():
                print(f"  {key}: {value}")
        else:
            print(f"{section}: {params}")
    print('----------------------------------------------')