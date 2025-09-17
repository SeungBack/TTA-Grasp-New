import wandb  # Import wandb
import datetime
import os
from omegaconf import OmegaConf

def init_wandb(cfg):
    """Initialize Weights & Biases."""
    # Generate a unique run name with timestamp
    run_name = f"{cfg.exp_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize wandb
    wandb.init(
        project=cfg.wandb.project_name,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        entity=cfg.wandb.entity,
        tags=[
            cfg.exp_name,
            cfg.tta_dataset.name,
            cfg.tta_dataset.split,
            cfg.tta_dataset.camera,
            cfg.eval_dataset.name,
            cfg.eval_dataset.split,
            cfg.eval_dataset.camera,
            cfg.tta.method if hasattr(cfg.tta, 'method') else "no_tta",
        ]
    )
    
    # Log the configuration as a YAML artifact
    cfg_artifact = wandb.Artifact(
        name=f"config_{run_name}", 
        type="config",
        description="Experiment configuration"
    )
    
    # Save config to temp file then add to artifact
    config_path = os.path.join(cfg.dump_dir, "config.yaml")
    os.makedirs(cfg.dump_dir, exist_ok=True)
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, f)
    
    cfg_artifact.add_file(config_path)
    wandb.log_artifact(cfg_artifact)
    
    return run_name
