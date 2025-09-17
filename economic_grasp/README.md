
## GRL
```
```
CUDA_VISIBLE_DEVICES=0 python train_grl.py --model economicgrasp --camera realsense --grl_camera realsense --grl_split test_seen --log_dir results/economicgrasp_grl_r2r_seen_0.1 --max_epoch 10 --batch_size 8 --dataset_root /home/seung/Datasets/GraspNet-1Billion/ --grl_use_ratio 0.1

CUDA_VISIBLE_DEVICES=1 python train_grl.py --model economicgrasp --camera realsense --grl_camera realsense --grl_split test_similar --log_dir results/economicgrasp_grl_r2r_similar_0.1 --max_epoch 10 --batch_size 8 --dataset_root /home/seung/Datasets/GraspNet-1Billion/ --grl_use_ratio 0.1

CUDA_VISIBLE_DEVICES=2 python train_grl.py --model economicgrasp --camera realsense --grl_camera realsense --grl_split test_novel --log_dir results/economicgrasp_grl_r2r_novel_0.1 --max_epoch 10 --batch_size 8 --dataset_root /home/seung/Datasets/GraspNet-1Billion/ --grl_use_ratio 0.1

CUDA_VISIBLE_DEVICES=3 python train_grl.py --model economicgrasp --camera realsense --grl_camera kinect --grl_split test_novel --log_dir results/economicgrasp_grl_r2k_novel_0.1 --max_epoch 10 --batch_size 8 --dataset_root /home/seung/Datasets/GraspNet-1Billion/ --grl_use_ratio 0.1




CUDA_VISIBLE_DEVICES=1 python test.py --checkpoint_path ckpts/economicgrasp_realsense.tar --camera kinect --dataset_root /home/seung/Datasets/GraspNet-1Billion/ --split test_seen --mode all --save_dir results/economicgrasp_r2k_seen_0.1

CUDA_VISIBLE_DEVICES=3 python test.py --checkpoint_path ckpts/economicgrasp_realsense.tar --camera kinect --dataset_root /home/seung/Datasets/GraspNet-1Billion/ --split test_novel --mode all --save_dir results/economicgrasp_r2k_novel_0.1





```

```


CUDA_VISIBLE_DEVICES=0 python main.py --cfg  configs/graspnet1b/notta/realsense_seen.yaml
CUDA_VISIBLE_DEVICES=1 python main.py --cfg  configs/graspnet1b/notta/realsense_similar.yaml
CUDA_VISIBLE_DEVICES=2 python main.py --cfg  configs/graspnet1b/notta/realsense_novel.yaml


CUDA_VISIBLE_DEVICES=0 python main.py --cfg  configs/graspnet1b/bn-adapt/realsense_similar_mixed_wl1.yaml
CUDA_VISIBLE_DEVICES=2 python main.py --cfg  configs/graspnet1b/bn-adapt/realsense_seen_mixed.yaml
CUDA_VISIBLE_DEVICES=3 python main.py --cfg  configs/graspnet1b/bn-adapt/realsense_novel_mixed.yaml


CUDA_VISIBLE_DEVICES=2 python main.py --cfg configs/graspnet1b/tta-grasp/debug_2.yaml
CUDA_VISIBLE_DEVICES=1 python main.py --cfg configs/graspnet1b/tta-grasp/debug.yaml




CUDA_VISIBLE_DEVICES=0 python train.py --model economicgrasp --camera realsense --log_dir results/debug --max_epoch 10 --batch_size 4 --dataset_root /home/seung/Workspaces/Datasets/GraspNet-1Billion


CUDA_VISIBLE_DEVICES=0 python test.py --model economicgrasp --camera realsense --log_dir results/debug --max_epoch 10 --batch_size 4 --dataset_root /home/seung/Workspaces/Datasets/GraspNet-1Billion



```c