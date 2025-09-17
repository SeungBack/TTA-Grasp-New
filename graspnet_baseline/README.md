## !TODO:
- [X] contrastive loss
- [X] weighted loss
- [X] Save weight
- [ ] Support EconomicGrasp
- [ ] Support BOP dataset
- [ ] large batch size
- [ ] Upper Bound

- [ ] reduce GPU memory for contrastive loss
- [ ] Ablate GPG loss

```
python main.py --config configs/base.yaml

CUDA_VISIBLE_DEVICES=0 python main.py --config configs/notta.yaml --opts dataset.split="test_seen_mixed_mini"
CUDA_VISIBLE_DEVICES=1 python main.py --config configs/notta.yaml --opts dataset.split="test_novel_mixed_mini"

CUDA_VISIBLE_DEVICES=0 python main_with_sweep.py --sweep --cfg configs/tta-grasp-seen-mixed-mini.yaml 

CUDA_VISIBLE_DEVICES=1 python main_with_sweep.py --swee --cfg configs/tta-grasp-similar-mixed-mini.yaml 
CUDA_VISIBLE_DEVICES=2 python main.py --sweep --cfg configs/tta-grasp-novel-mixed-mini.yaml 


CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/graspnet1b/tta-grasp/exp_bs1/realsense_similar.yaml 
CUDA_VISIBLE_DEVICES=1 python main.py --cfg configs/graspnet1b/tta-grasp/exp_bs1/realsense_similar_hflip.yaml 
CUDA_VISIBLE_DEVICES=2 python main.py --cfg configs/graspnet1b/tta-grasp/exp_bs1/realsense_similar_nolamda.yaml 
CUDA_VISIBLE_DEVICES=2 python main.py --cfg configs/graspclutter6d/tta-grasp/realsense_mini_hflip_nolamda.yaml 

CUDA_VISIBLE_DEVICES=2 python main.py --cfg configs/graspnet1b/tta-grasp/exp_tmp/realsense_similar.yaml 

CUDA_VISIBLE_DEVICES=3 python main.py --cfg  configs/graspnet1b/cotta/realsense_similar_hflip.yaml


CUDA_VISIBLE_DEVICES=2 python main.py --cfg  configs/graspnet1b/tta-grasp/exp_bs1/realsense_similar_mixed_v2.yaml
CUDA_VISIBLE_DEVICES=1 python main.py --cfg  configs/graspnet1b/tta-grasp/exp_bs1/realsense_similar_mixed_noconst_256_0.3.yaml
CUDA_VISIBLE_DEVICES=2 python main.py --cfg  configs/graspnet1b/tta-grasp/exp_bs1/realsense_similar_mixed_noconst_256.yaml

CUDA_VISIBLE_DEVICES=3 python main.py --cfg  configs/graspnet1b/bn-adapt/bs1/realsense_similar_mixed_mini.yaml

CUDA_VISIBLE_DEVICES=3 python main.py --cfg  configs/graspnet1b/tta-grasp/exp_tmp/realsense_similar_mixed_mini_.yaml


CUDA_VISIBLE_DEVICES=0 python main.py --cfg  configs/graspnet1b/tta-grasp/exp_tmp/realsense_similar_mixed_mini_kd.yaml



CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/graspnet1b/tta-grasp/realsense_similar_mixed.yaml

CUDA_VISIBLE_DEVICES=1 python main.py --cfg configs/graspnet1b/bn-adapt/realsense_similar_mixed.yaml 
CUDA_VISIBLE_DEVICES=2 python main.py --cfg configs/graspnet1b/notta/realsense_seen.yaml 
CUDA_VISIBLE_DEVICES=3 python main.py --cfg configs/graspnet1b/bn-adapt/realsense_similar_mixed_wl16.yaml 

CUDA_VISIBLE_DEVICES=3 python main.py --cfg configs/graspnet1b/tta-grasp/realsense_similar_mixed_mini.yaml 


CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/graspnet1b/tta-grasp/realsense_similar_mixed_q0.3_u-1.yaml
CUDA_VISIBLE_DEVICES=1 python main.py --cfg configs/graspnet1b/tta-grasp/realsense_similar_mixed_q0.4_u0.05.yaml
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/graspnet1b/tta-grasp/realsense_similar_mixed_q0.4_u0.1_g1b.yaml

CUDA_VISIBLE_DEVICES=2 python main.py --cfg configs/graspnet1b/tta-grasp/realsense_similar_mixed_q0.5_u-1_full.yaml
CUDA_VISIBLE_DEVICES=3 python main.py --cfg configs/graspnet1b/tta-grasp/realsense_similar_mixed_q0.5_u0.05_full.yaml




```n