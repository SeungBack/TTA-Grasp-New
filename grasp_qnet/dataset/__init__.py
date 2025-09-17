
def get_dataset(cfg, dataset_name, split, camera):
    if dataset_name == 'graspnet1b':
        # Init datasets and dataloaders
        test_dataset = GraspNetDataset(
            cfg.graspnet1b.root, 
            camera=camera, 
            split=split,
            voxel_size=cfg.model.voxel_size,
            num_points=cfg.model.num_point, 
            remove_outlier=True, 
            augment=False, 
            load_label=False, 
            return_raw_cloud=cfg.tta.method in ['cotta', 'tta-grasp']
        )
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=cfg.tta.batch_size, 
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn
        )
    elif dataset_name == 'graspclutter6d':
        test_dataset = GraspClutter6DDataset(
            cfg.graspclutter6d.root, 
            camera=camera, 
            split=split,
            voxel_size=cfg.graspclutter6d.voxel_size,
            num_points=cfg.model.num_point, 
            remove_outlier=True, 
            augment=False, 
            load_label=False, 
            return_raw_cloud=cfg.tta.method in ['cotta', 'tta-grasp']
        )
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=cfg.tta.batch_size, 
            shuffle=False,
            num_workers=cfg.num_workers, 
            collate_fn=collate_fn
        )
    else:
        raise ValueError(f'Invalid dataset: {dataset_name}')
    return test_dataset, test_dataloader