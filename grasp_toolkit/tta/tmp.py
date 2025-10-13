# 1. TTA_Grasp_EconomicGrasp.forward_and_adapt 수정

def forward_and_adapt(self, batch_data):
    
    self.optimizer.zero_grad()
    batch_size = batch_data['point_clouds'].shape[0]
    merged_grasp_preds_ema = [[] for _ in range(batch_size)]
    merged_grasp_top_view_inds_ema = [[] for _ in range(batch_size)]
    merged_grasp_angles_ema = [[] for _ in range(batch_size)]
    merged_mat_aug = [[] for _ in range(batch_size)]
    grasp_preds_ensembles = [[] for _ in range(batch_size)]
    merged_grasp_q = [[] for _ in range(batch_size)]
    merged_view_scores_ema = [[] for _ in range(batch_size)]
    merged_graspness_scores_ema = [[] for _ in range(batch_size)]
    merged_objectness_scores_ema = [[] for _ in range(batch_size)]
    merged_confidence_weights = [[] for _ in range(batch_size)]  # 새로 추가
    
    # teacher EMA model
    with torch.no_grad():
        for aug_type in self.cfg.tta.aug_types.split(','):
            # Forward pass with ema model
            pc_aug, mat_aug = augment_cloud(batch_data['point_clouds'], type=aug_type)
            
            end_points_ema = self.model_ema({
                'point_clouds': pc_aug, 
                'coordinates_for_voxel': pc_aug / self.cfg.model.voxel_size}
                )
            # [1, 1024, 17], [1, 1024]
            grasp_preds_raw_ema, grasp_angles, view_scores, graspness_score, objectness_score = self.pred_decode_raw(end_points_ema)
            
            # get grasp preds
            if aug_type == 'none':
                grasp_preds = grasp_preds_raw_ema
                
            for b in range(batch_size):
                gg_array_ema = grasp_preds_raw_ema[b]
                scene_cloud = batch_data['point_clouds_raw'][b]
                scene_cloud = transform_point_cloud(scene_cloud, mat_aug)

                # prepare object and gripper clouds for ALL grasps
                obj_clouds, gripper_clouds, gg_array_all = crop_inner_cloud(
                    gg_array_ema.clone(), scene_cloud, self.cfg.tta.geval_net.min_points,
                )
                if len(obj_clouds) == 0:
                    print("No valid grasp predictions (crop_inner_cloud returned empty)")
                    return grasp_preds, None
                
                # run geval net in batch-wise manner for ALL grasps
                grasp_q = []
                uncertainty = []
                n_items = len(obj_clouds)
                geval_bs = self.cfg.tta.geval_net.batch_size
                for batch_idx in range(math.ceil(n_items / geval_bs)):
                    start_idx = batch_idx * geval_bs
                    end_idx = min(start_idx + geval_bs, n_items)
                    obj_clouds_batch = obj_clouds[start_idx:end_idx]
                    gripper_clouds_batch = gripper_clouds[start_idx:end_idx]

                    # use mc_dropout if enabled                            
                    if self.cfg.tta.geval_net.uncertainty_thresh > 0.0:
                        grasp_q_batch, std_batch = self.geval_net.forward_mc_dropout(
                            obj_clouds_batch, gripper_clouds_batch, 
                            N=self.cfg.tta.geval_net.uncertainty_n)
                        uncertainty.extend(std_batch)
                    else:
                        grasp_q_batch = self.geval_net(obj_clouds_batch, gripper_clouds_batch)
                    grasp_q.extend(grasp_q_batch)
                grasp_q = torch.cat(grasp_q, dim=0)

                if grasp_q.numel() == 0 or grasp_q.dim() == 0:
                    print("No valid grasp predictions (after grasp_q check)")
                    return grasp_preds, None
                
                # 모든 grasp 사용 - 아무 threshold 없음
                gg_array_filt = gg_array_all
                
                # Grasp quality score에 직접 비례해서 confidence weighting
                confidence_weights = grasp_q  # 0~1 범위의 quality score 그대로 사용
                
                # After processing, get indices for EMA data BEFORE modifying scores
                idx_in_ema = torch.tensor(get_index_A_to_B(gg_array_filt.clone(), gg_array_ema.clone()), 
                                        device=gg_array_ema.device)
                
                # Quality를 grasp score 조정에도 사용할지 결정
                if hasattr(self.cfg.tta, 'adjust_grasp_scores') and self.cfg.tta.adjust_grasp_scores:
                    # Option 2: Quality로 grasp score 조정 (EMA 데이터에도 반영)
                    baseline_score = getattr(self.cfg.tta, 'baseline_score', 0.5)
                    # 가중평균: quality * teacher_score + (1 - quality) * baseline
                    teacher_scores = gg_array_filt[:, 0]  # grasp scores
                    quality_adjusted_scores = grasp_q * teacher_scores + (1 - grasp_q) * baseline_score
                    gg_array_filt[:, 0] = quality_adjusted_scores  # ensemble용 업데이트
                    
                    # EMA 데이터도 동일하게 조정
                    ema_teacher_scores = grasp_preds_raw_ema[b][idx_in_ema][:, 0]
                    ema_quality_adjusted = grasp_q * ema_teacher_scores + (1 - grasp_q) * baseline_score
                    adjusted_ema_preds = grasp_preds_raw_ema[b][idx_in_ema].clone()
                    adjusted_ema_preds[:, 0] = ema_quality_adjusted
                    merged_grasp_preds_ema[b].append(adjusted_ema_preds)
                else:
                    # Option 1: Teacher score 그대로 사용
                    merged_grasp_preds_ema[b].append(grasp_preds_raw_ema[b][idx_in_ema])
                
                # Store other EMA predictions with corresponding indices
                merged_grasp_top_view_inds_ema[b].append(end_points_ema['grasp_top_view_inds'][b][idx_in_ema])
                merged_grasp_angles_ema[b].append(grasp_angles[b][idx_in_ema])
                merged_view_scores_ema[b].append(view_scores[b][idx_in_ema])
                merged_confidence_weights[b].append(confidence_weights)
                merged_grasp_q[b].append(grasp_q)
                merged_mat_aug[b].extend([mat_aug] * gg_array_filt.shape[0])
                merged_graspness_scores_ema[b].append(graspness_score)
                merged_objectness_scores_ema[b].append(objectness_score)
                
                print(f"Using ALL {len(gg_array_filt)} grasps with quality-based confidence weighting")
                    
                if gg_array_filt.shape[0] == 0:
                    print("No valid grasp predictions (after processing)")
                    return grasp_preds, None
                    
                grasp_preds_ensembles[b].append(gg_array_filt.clone())

        # merge all the grasp predictions from different augmentations
        for b in range(len(merged_grasp_preds_ema)):
            merged_grasp_preds_ema[b] = torch.cat(merged_grasp_preds_ema[b], dim=0)
            merged_grasp_top_view_inds_ema[b] = torch.cat(merged_grasp_top_view_inds_ema[b], dim=0)
            merged_grasp_angles_ema[b] = torch.cat(merged_grasp_angles_ema[b], dim=0)
            merged_view_scores_ema[b] = torch.cat(merged_view_scores_ema[b], dim=0)
            merged_confidence_weights[b] = torch.cat(merged_confidence_weights[b], dim=0)  # 새로 추가
            merged_graspness_scores_ema[b] = torch.stack(merged_graspness_scores_ema[b], dim=0).mean(dim=0)
            merged_objectness_scores_ema[b] = torch.stack(merged_objectness_scores_ema[b], dim=0).mean(dim=0)
            
        merged_graspness_scores_ema = torch.cat(merged_graspness_scores_ema, dim=0)
        merged_objectness_scores_ema = torch.cat(merged_objectness_scores_ema, dim=0)
                         
        batch_data['batch_grasp_preds_ema'] = merged_grasp_preds_ema
        batch_data['batch_grasp_top_view_inds_ema'] = merged_grasp_top_view_inds_ema
        batch_data['batch_grasp_angles_ema'] = merged_grasp_angles_ema
        batch_data['mat_aug'] = merged_mat_aug
        batch_data['batch_view_scores_ema'] = merged_view_scores_ema
        batch_data['batch_confidence_weights_ema'] = merged_confidence_weights  # 새로 추가
        batch_data['graspness_label'] = merged_graspness_scores_ema
        batch_data['objectness_label'] = torch.argmax(merged_objectness_scores_ema, dim=1)
        
    
    # merge all the grasp predictions from different ensembles
    grasp_preds = [np.zeros((0, 17)) for _ in range(batch_size)]
    for b in range(len(grasp_preds_ensembles)):
        for i in range(len(grasp_preds_ensembles[b])):
            gg = GraspGroup(grasp_preds_ensembles[b][i].cpu().numpy())
            H = np.eye(4)
            H[:3, :3] = get_aug_matrix(self.cfg.tta.aug_types.split(',')[i])
            gg = gg.transform(H)
            grasp_preds[b] = np.concatenate([grasp_preds[b], gg.grasp_group_array], axis=0)
    
    num_grasps = len(grasp_preds[0])
        
    if num_grasps > self.cfg.tta.min_grasps:
        # student model
        end_points = self.model(batch_data)
        end_points['loss_type'] = self.cfg.tta.loss_type
        loss, end_points = self.compute_tta_loss(end_points)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Update teacher model with EMA
        self.model_ema = ema_update_model(self.model_ema, self.model, self.cfg.tta.ema_ratio, self.device)
        if self.cfg.tta.rst_ratio > 0:
            self.stochastic_restore(self.model, self.model_states)
    else:
        end_points = {}

    return grasp_preds, end_points


# 2. label_generation.py의 process_grasp_pseudo_label 수정

def process_grasp_pseudo_label(end_points):
    """ Process labels according to scene points and object poses. """
    seed_xyzs = end_points['xyz_graspable']
    pred_top_view_inds = end_points['grasp_top_view_inds']
    batch_size, num_samples, _ = seed_xyzs.size()

    valid_points_count = 0
    valid_views_count = 0

    batch_grasp_points = []
    batch_grasp_views_rot = []
    batch_view_graspness = []
    batch_grasp_rotations = []
    batch_grasp_depth = []
    batch_grasp_scores = []
    batch_grasp_widths = []
    batch_valid_mask = []
    batch_confidence_weights = []  # 새로 추가
    
    for i in range(batch_size):
        seed_xyz = seed_xyzs[i]
        pred_top_view = pred_top_view_inds[i]
    
        gg_array_ema = end_points['batch_grasp_preds_ema'][i]
        grasp_top_view_inds_ema = end_points['batch_grasp_top_view_inds_ema'][i]
        grasp_angles_ema = end_points['batch_grasp_angles_ema'][i]
        mat_aug = end_points['mat_aug'][i]
        view_scores_ema = end_points['batch_view_scores_ema'][i]
        confidence_weights_ema = end_points['batch_confidence_weights_ema'][i]  # 새로 추가

        # get merged grasp points for label computation
        # transform the view from object coordinate system to scene coordinate system
        grasp_points_merged = []
        grasp_views_rot_merged = []
        grasp_rotations_merged = []
        grasp_depth_merged = []
        grasp_scores_merged = []
        grasp_widths_merged = []
        view_graspness_merged = []
        top_view_index_merged = []

        N = len(gg_array_ema)
        V, A, D = cfgs.num_view, 12, 4
        device = seed_xyz.device
        
        grasp_scores_all = gg_array_ema[:, 0]  # (N,) - 이미 quality adjustment 적용됨
        grasp_widths_all = gg_array_ema[:, 1]  # (N,)
        grasp_depths_all = gg_array_ema[:, 3]  # (N,)
        grasp_points_all = gg_array_ema[:, 13:16]  # (N, 3)
        
        # Create pose matrices for all objects
        poses = torch.eye(4, device=device).unsqueeze(0).repeat(N, 1, 1)  # (N, 4, 4)
        for j, mat in enumerate(mat_aug):
            if mat is not None:
                poses[j, :3, :3] = mat
        
        # Initialize grasp tensors for all predictions
        grasp_scores = torch.zeros(N, V, A, D, dtype=torch.float32, device=device)
        grasp_widths = torch.zeros(N, V, A, D, dtype=torch.float32, device=device)
        view_graspness = view_scores_ema  # (N, V) - already in correct format

        # Compute indices for all predictions
        angle_inds = (grasp_angles_ema * A / torch.pi).long()  # (N,)
        depth_inds = ((grasp_depths_all / 0.01) - 1).long()  # (N,)

        # Fill grasp tensors using vectorized indexing
        batch_inds = torch.arange(N, device=device)
        grasp_scores[batch_inds, grasp_top_view_inds_ema, angle_inds, depth_inds] = grasp_scores_all
        grasp_widths[batch_inds, grasp_top_view_inds_ema, angle_inds, depth_inds] = grasp_widths_all

        # Vectorized depth and angle selection (same as original logic)
        grasp_score_label_max_depth, grasp_score_label_max_depth_idx = grasp_scores.max(-1)
        grasp_widths = grasp_widths.gather(-1, grasp_score_label_max_depth_idx.unsqueeze(-1)).squeeze(-1)
        
        grasp_score_label_max_angle, grasp_score_label_max_angle_idx = grasp_score_label_max_depth.max(-1)
        grasp_depths = grasp_score_label_max_depth_idx.gather(-1, grasp_score_label_max_angle_idx.unsqueeze(-1)).squeeze(-1)
        grasp_rotations = grasp_score_label_max_angle_idx  # [N, V]
        grasp_scores = grasp_score_label_max_angle  # [N, V]
        grasp_widths = grasp_widths.gather(-1, grasp_score_label_max_angle_idx.unsqueeze(-1)).squeeze(-1)

        # Select top views for all predictions
        values, top_view_index = torch.topk(view_graspness, k=V)
        grasp_rotations = torch.gather(grasp_rotations, 1, top_view_index)
        grasp_depths = torch.gather(grasp_depths, 1, top_view_index)
        grasp_scores = torch.gather(grasp_scores, 1, top_view_index)
        grasp_widths = torch.gather(grasp_widths, 1, top_view_index)

        # Generate template views (shared across all predictions)
        grasp_views = generate_grasp_views(V).to(device)  # [V, 3]

        # Transform grasp points for all predictions
        grasp_points_trans_list = []
        for n in range(N):
            pts_trans = transform_point_cloud(grasp_points_all[n:n+1], poses[n], '3x4')
            grasp_points_trans_list.append(pts_trans)
        grasp_points_trans = torch.cat(grasp_points_trans_list, dim=0)

        # Transform grasp views for all predictions  
        grasp_views_expanded = grasp_views.unsqueeze(0).expand(N, -1, -1)  # (N, V, 3)
        grasp_views_trans = torch.bmm(poses[:, :3, :3], grasp_views_expanded.transpose(1, 2)).transpose(1, 2)  # (N, V, 3)

        # Generate view rotation matrices for all predictions
        angles = torch.zeros(N, V, dtype=grasp_views.dtype, device=device)
        try:
            # Try batch operation first
            grasp_views_rot = batch_viewpoint_params_to_matrix(-grasp_views_expanded, angles)  # (N, V, 3, 3)
        except:
            # Fallback to sequential processing if batch not supported
            grasp_views_rot = torch.stack([
                batch_viewpoint_params_to_matrix(-grasp_views_expanded[n], angles[n]) 
                for n in range(N)
            ], dim=0)

        # Transform rotation matrices
        grasp_views_rot_trans = torch.matmul(poses[:, :3, :3].unsqueeze(1), grasp_views_rot)  # (N, V, 3, 3)

        # ================== VIEW ASSIGNMENT (KNN MATCHING) ==================
        
        # Prepare reference views (same for all predictions)
        grasp_views_ref = grasp_views.transpose(0, 1).contiguous()  # (3, V)
        
        # Process each prediction's view assignment (KNN requires individual processing)
        view_graspness_trans_list = []
        grasp_views_rot_trans_reordered_list = []
        top_view_index_trans_list = []
        
        for n in range(N):
            # Transform views for this prediction
            grasp_views_trans_n = grasp_views_trans[n].transpose(0, 1).contiguous().unsqueeze(0)  # (1, 3, V)
            grasp_views_ref_expanded = grasp_views_ref.unsqueeze(0)  # (1, 3, V)
            
            # Find nearest neighbor views
            view_inds_raw = knn(grasp_views_trans_n, grasp_views_ref_expanded, k=1).squeeze().squeeze()
            view_inds = view_inds_raw - 1  # [V]
            
            # Transform view graspness
            view_graspness_trans = torch.index_select(view_graspness[n:n+1], 1, view_inds)  # [1, V]
            
            # Transform rotation matrices
            grasp_views_rot_trans_n = torch.index_select(grasp_views_rot_trans[n], 0, view_inds)
            grasp_views_rot_trans_n = grasp_views_rot_trans_n.unsqueeze(0)  # [1, V, 3, 3]
            
            # Create top view index mapping
            top_view_index_trans = (-1 * torch.ones((1, grasp_rotations.shape[1]), dtype=torch.long, device=device))
            view_inds_3d = view_inds.unsqueeze(0).unsqueeze(-1)  # [1, V, 1] 
            top_view_3d = top_view_index[n].unsqueeze(0).unsqueeze(0)  # [1, 1, V]
            matches = (view_inds_3d == top_view_3d)  # [1, V, V] - 3D tensor
            tpid, tvip, tids = torch.where(matches)  # Now returns 3 values as expected
            if len(tids) > 0:
                top_view_index_trans[tpid, tvip] = tids
            
            view_graspness_trans_list.append(view_graspness_trans)
            grasp_views_rot_trans_reordered_list.append(grasp_views_rot_trans_n)
            top_view_index_trans_list.append(top_view_index_trans)

        # Combine results from all predictions
        grasp_points_merged = grasp_points_trans  # [N, 3]
        view_graspness_merged = torch.cat(view_graspness_trans_list, dim=0)  # [N, V]
        top_view_index_merged = torch.cat(top_view_index_trans_list, dim=0)  # [N, V]
        grasp_rotations_merged = grasp_rotations.to(torch.int32)  # [N, V]
        grasp_depth_merged = grasp_depths.to(torch.int32)  # [N, V]
        grasp_scores_merged = grasp_scores  # [N, V]
        grasp_widths_merged = grasp_widths  # [N, V]
        grasp_views_rot_merged = torch.cat(grasp_views_rot_trans_reordered_list, dim=0)  # [N, V, 3, 3]
        # ================== ASSIGNMENT TO SCENE POINTS (KNN MATCHING) =================

        # compute nearest neighbors
        seed_xyz_ = seed_xyz.transpose(0, 1).contiguous().unsqueeze(0)
        grasp_points_merged_ = grasp_points_merged.transpose(0, 1).contiguous().unsqueeze(0)
        nn_inds = knn(grasp_points_merged_, seed_xyz_, k=1).squeeze() - 1

        # assign anchor points to real points
        grasp_points_merged = torch.index_select(grasp_points_merged, 0, nn_inds)
        # [1024 (scene points after sample), 3]
        grasp_views_rot_merged = torch.index_select(grasp_views_rot_merged, 0, nn_inds)
        # [1024 (scene points after sample), 300, 3, 3]
        view_graspness_merged = torch.index_select(view_graspness_merged, 0, nn_inds)
        # [1024 (scene points after sample), 300]
        top_view_index_merged = torch.index_select(top_view_index_merged, 0, nn_inds)
        # [1024 (scene points after sample), num_of_view]
        grasp_rotations_merged = torch.index_select(grasp_rotations_merged, 0, nn_inds)
        # [1024 (scene points after sample), num_of_view]
        grasp_depth_merged = torch.index_select(grasp_depth_merged, 0, nn_inds)
        # [1024 (scene points after sample), num_of_view]
        grasp_scores_merged = torch.index_select(grasp_scores_merged, 0, nn_inds)
        # [1024 (scene points after sample), num_of_view]
        grasp_widths_merged = torch.index_select(grasp_widths_merged, 0, nn_inds)
        # [1024 (scene points after sample), num_of_view]

        # confidence weight도 동일하게 변환
        confidence_weights_merged = torch.index_select(confidence_weights_ema, 0, nn_inds)
        # [1024 (scene points after sample)]

        # select top view's rot, score and width
        # we only assign labels when the pred view is in the pre-defined 60 top view, others are zero
        pred_top_view_ = pred_top_view.view(num_samples, 1, 1, 1).expand(-1, -1, 3, 3)
        # [1024 (points after sample), 1, 3, 3]
        top_grasp_views_rot = torch.gather(grasp_views_rot_merged, 1, pred_top_view_).squeeze(1)
        # [1024 (points after sample), 3, 3]
        pid, vid = torch.where(pred_top_view.unsqueeze(-1) == top_view_index_merged)
        # both pid and vid are [true numbers], where(condition) equals to nonzero(condition)
        top_grasp_rotations = 12 * torch.ones(num_samples, dtype=torch.int32).to(seed_xyz.device)
        # [1024 (points after sample)]
        top_grasp_depth = 4 * torch.ones(num_samples, dtype=torch.int32).to(seed_xyz.device)
        # [1024 (points after sample)]
        top_grasp_scores = torch.zeros(num_samples, dtype=torch.float32).to(seed_xyz.device)
        # [1024 (points after sample)]
        top_grasp_widths = 0.1 * torch.ones(num_samples, dtype=torch.float32).to(seed_xyz.device)
        # [1024 (points after sample)]
        top_confidence_weights = torch.ones(num_samples, dtype=torch.float32).to(seed_xyz.device) * 0.1
        # [1024 (points after sample)] - default low confidence
        
        top_grasp_rotations[pid] = torch.gather(grasp_rotations_merged[pid], 1, vid.view(-1, 1)).squeeze(1)
        top_grasp_depth[pid] = torch.gather(grasp_depth_merged[pid], 1, vid.view(-1, 1)).squeeze(1)
        top_grasp_scores[pid] = torch.gather(grasp_scores_merged[pid], 1, vid.view(-1, 1)).squeeze(1)  # 이미 quality adjusted
        top_grasp_widths[pid] = torch.gather(grasp_widths_merged[pid], 1, vid.view(-1, 1)).squeeze(1)
        top_confidence_weights[pid] = confidence_weights_merged[pid]

        # only compute loss in the points with correct matching (so compute the mask first)
        dist = compute_pointwise_dists(seed_xyz, grasp_points_merged)
        valid_point_mask = dist < 0.005
        valid_view_mask = torch.zeros(num_samples, dtype=torch.bool).to(seed_xyz.device)
        valid_view_mask[pid] = True
        valid_points_count = valid_points_count + torch.sum(valid_point_mask)
        valid_views_count = valid_views_count + torch.sum(valid_view_mask)
        valid_score_mask = top_grasp_scores > 0.0
        valid_mask = valid_score_mask  & valid_view_mask  & valid_point_mask

        # add to batch
        batch_grasp_points.append(grasp_points_merged)
        batch_grasp_views_rot.append(top_grasp_views_rot)
        batch_view_graspness.append(view_graspness_merged)
        batch_grasp_rotations.append(top_grasp_rotations)
        batch_grasp_depth.append(top_grasp_depth)
        batch_grasp_scores.append(top_grasp_scores)
        batch_grasp_widths.append(top_grasp_widths)
        batch_valid_mask.append(valid_mask)
        batch_confidence_weights.append(top_confidence_weights)  # 새로 추가

    batch_grasp_points = torch.stack(batch_grasp_points, 0)
    # [B (batch size), 1024 (scene points after sample), 3]
    batch_grasp_views_rot = torch.stack(batch_grasp_views_rot, 0)
    # [B (batch size), 1024 (scene points after sample), 3, 3]
    batch_view_graspness = torch.stack(batch_view_graspness, 0)
    # [B (batch size), 1024 (scene points after sample), 300]
    batch_grasp_rotations = torch.stack(batch_grasp_rotations, 0)
    # [B (batch size), 1024 (scene points after sample)]
    batch_grasp_depth = torch.stack(batch_grasp_depth, 0) # 0~4 
    # [B (batch size), 1024 (scene points after sample)]
    batch_grasp_scores = torch.stack(batch_grasp_scores, 0)
    # [B (batch size), 1024 (scene points after sample)]
    batch_grasp_widths = torch.stack(batch_grasp_widths, 0)
    # [B (batch size), 1024 (scene points after sample)]
    batch_valid_mask = torch.stack(batch_valid_mask, 0)
    # [B (batch size), 1024 (scene points after sample)]
    batch_confidence_weights = torch.stack(batch_confidence_weights, 0)
    # [B (batch size), 1024 (scene points after sample)]
    
    end_points['batch_grasp_point'] = batch_grasp_points
    end_points['batch_grasp_rotations'] = batch_grasp_rotations
    end_points['batch_grasp_depth'] = batch_grasp_depth
    end_points['batch_grasp_score'] = batch_grasp_scores
    end_points['batch_grasp_width'] = batch_grasp_widths
    end_points['batch_grasp_view_graspness'] = batch_view_graspness
    end_points['batch_valid_mask'] = batch_valid_mask
    end_points['batch_confidence_weights'] = batch_confidence_weights  # 새로 추가
    end_points['C: Valid Points'] = valid_points_count / batch_size
    return batch_grasp_views_rot, end_points


# 3. loss_economicgrasp.py 수정 - 모든 loss 함수에 confidence weighting 적용

def compute_angle_loss(end_points):
    criterion = nn.CrossEntropyLoss(reduction='none')
    grasp_angle_pred = end_points['grasp_angle_pred']
    grasp_angle_label = end_points['batch_grasp_rotations'].long()
    valid_mask = end_points['batch_valid_mask']
    confidence_weights = end_points.get('batch_confidence_weights', torch.ones_like(valid_mask.float()))
    
    loss = criterion(grasp_angle_pred, grasp_angle_label)
    if torch.sum(valid_mask) == 0:
        loss = 0 * torch.sum(loss)
    else:
        # Confidence weighting 적용
        weighted_loss = loss * confidence_weights
        loss = weighted_loss[valid_mask].mean()
    
    end_points['B: Angle Loss'] = loss
    end_points['D: Angle Acc'] = (torch.argmax(grasp_angle_pred, 1) == grasp_angle_label)[valid_mask].float().mean()
    return loss, end_points

def compute_depth_loss(end_points):
    criterion = nn.CrossEntropyLoss(reduction='none')
    grasp_depth_pred = end_points['grasp_depth_pred']
    grasp_depth_label = end_points['batch_grasp_depth'].long()
    valid_mask = end_points['batch_valid_mask']
    confidence_weights = end_points.get('batch_confidence_weights', torch.ones_like(valid_mask.float()))
    
    loss = criterion(grasp_depth_pred, grasp_depth_label)
    if torch.sum(valid_mask) == 0:
        loss = 0 * torch.sum(loss)
    else:
        weighted_loss = loss * confidence_weights
        loss = weighted_loss[valid_mask].mean()
    
    end_points['B: Depth Loss'] = loss
    end_points['D: Depth Acc'] = (torch.argmax(grasp_depth_pred, 1) == grasp_depth_label)[valid_mask].float().mean()
    return loss, end_points

def compute_score_loss_cls(end_points):
    criterion = nn.CrossEntropyLoss(reduction='none')
    grasp_score_pred = end_points['grasp_score_pred']
    grasp_score_label = (end_points['batch_grasp_score'] * 10 / 2).long()
    valid_mask = end_points['batch_valid_mask']
    confidence_weights = end_points.get('batch_confidence_weights', torch.ones_like(valid_mask.float()))
    
    loss = criterion(grasp_score_pred.squeeze(1), grasp_score_label)
    if torch.sum(valid_mask) == 0:
        loss = 0 * torch.sum(loss)
    else:
        weighted_loss = loss * confidence_weights
        loss = weighted_loss[valid_mask].mean()
    
    end_points['B: Score Loss'] = loss
    end_points['D: Score Acc'] = (torch.argmax(grasp_score_pred, 1) == grasp_score_label)[valid_mask].float().mean()
    return loss, end_points

def compute_width_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='none')
    grasp_width_pred = end_points['grasp_width_pred']
    grasp_width_label = end_points['batch_grasp_width'] * 10
    valid_mask = end_points['batch_valid_mask']
    confidence_weights = end_points.get('batch_confidence_weights', torch.ones_like(valid_mask.float()))
    
    loss = criterion(grasp_width_pred.squeeze(1), grasp_width_label)
    if torch.sum(valid_mask) == 0:
        loss = 0 * torch.sum(loss)
    else:
        weighted_loss = loss * confidence_weights
        loss = weighted_loss[valid_mask].mean()
    
    end_points['B: Width Loss'] = loss
    return loss, end_points

def compute_view_graspness_tta_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='none')
    view_score = end_points['view_score']
    view_label = end_points['batch_grasp_view_graspness']
    valid_mask = end_points['batch_valid_mask']
    confidence_weights = end_points.get('batch_confidence_weights', torch.ones_like(valid_mask.float()))
    
    loss = criterion(view_score, view_label)
    if torch.sum(valid_mask) == 0:
        loss = 0 * torch.sum(loss)
    else:
        # view는 2D이므로 confidence weight 차원 맞춤
        weighted_loss = loss * confidence_weights.unsqueeze(-1).expand_as(loss)
        loss = weighted_loss[valid_mask].mean()
    
    end_points['B: View Loss'] = loss
    return loss, end_points

def compute_graspness_tta_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='none')
    graspness_score = end_points['graspness_score'].squeeze(1)
    graspness_label = end_points['graspness_label'].squeeze(-1)
    loss_mask = end_points['objectness_label'].bool()
    confidence_weights = end_points.get('batch_confidence_weights', torch.ones_like(loss_mask.float()))
    
    loss = criterion(graspness_score, graspness_label)
    if torch.sum(loss_mask) == 0:
        loss = 0 * torch.sum(loss)
    else:
        # Apply confidence weighting
        weighted_loss = loss * confidence_weights
        loss = weighted_loss[loss_mask].mean()

    end_points['B: Graspness Loss'] = loss
    return loss, end_points

def compute_objectness_loss(end_points):
    criterion = nn.CrossEntropyLoss(reduction='none')
    objectness_score = end_points['objectness_score']
    objectness_label = end_points['objectness_label']
    confidence_weights = end_points.get('batch_confidence_weights', torch.ones_like(objectness_label.float()))
    
    loss = criterion(objectness_score, objectness_label)
    # Apply confidence weighting
    weighted_loss = loss * confidence_weights
    loss = weighted_loss.mean()
    
    end_points['B: Objectness Loss'] = loss
    objectness_pred = torch.argmax(objectness_score, 1)
    end_points['D: Objectness Acc'] = (objectness_pred == objectness_label.long()).float().mean()
    return loss, end_points


# 4. Config 설정 예시
"""
# Option 1: Confidence weighting만 사용 (추천)
cfg.tta.adjust_grasp_scores = False

# Option 2: Grasp score도 quality에 따라 조정
cfg.tta.adjust_grasp_scores = True
cfg.tta.baseline_score = 0.5

# 기타 설정
cfg.tta.geval_net.min_points = 50    # crop_inner_cloud 최소 points
cfg.tta.min_grasps = 10              # 최소 grasp 개수
"""