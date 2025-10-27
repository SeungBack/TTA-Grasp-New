

conda activate testadagrasp 
conda activate tta-grasp
tensorboard --logdir logs

# train







CUDA_VISIBLE_DEVICES=0,1 python main_reg.py --net dgcnn --batch_size 512 --learning_rate 1e-3 --min_learning_rate 1e-5 --log_dir logs/lr_1e-3_adam_bs512_decay_new_aug --save_best_only --evals_per_epoch 2 --dropout 0.1 --net dgcnn

CUDA_VISIBLE_DEVICES=2,3 python main_reg.py --net dgcnn --batch_size 256 --learning_rate 1e-3 --min_learning_rate 1e-5 --log_dir logs/lr_1e-3_adam_bs256_decay_new_aug --save_best_only --evals_per_epoch 2 --dropout 0.1 --net dgcnn




## test normal


# test
CUDA_VISIBLE_DEVICES=0 python test.py --net dgcnn --ckpt_path logs/dgcnn_lr1e-4_synreal_final_bs128/checkpoint_epoch1.tar --test_split test_similar

CUDA_VISIBLE_DEVICES=1 python test.py --net dgcnn --ckpt_path logs/dgcnn_lr1e-4_synreal_final_do_bs256_sigmoid/checkpoint_epoch2.tar --test_split test_similar

CUDA_VISIBLE_DEVICES=2 python test.py --net dgcnn --ckpt_path logs/dgcnn_lr1e-4_synreal_final_do_bs256_sigmoid/checkpoint_epoch3.tar --test_split test_similar

CUDA_VISIBLE_DEVICES=3 python test.py --net dgcnn --ckpt_path logs/dgcnn_lr1e-4_synreal_final_do_bs256_sigmoid/checkpoint_epoch4.tar --test_split test_similar


CUDA_VISIBLE_DEVICES=3 python test.py --net dgcnn --ckpt_path /home/seung/Workspaces/grasp/TestAdaGrasp/grasp_qnet/ckpts/gevalnet-dgcnn.tar --test_split test_similar


CUDA_VISIBLE_DEVICES=0 python test.py --net dgcnn --ckpt_path ckpts/graspqnet-g1b-acr.tar --test_split test_seen --mc_dropout
CUDA_VISIBLE_DEVICES=1 python test.py --net dgcnn --ckpt_path ckpts/graspqnet-g1b-acr.tar --test_split test_similar --mc_dropout
CUDA_VISIBLE_DEVICES=2 python test.py --net dgcnn --ckpt_path ckpts/graspqnet-g1b-acr.tar --test_split test_novel --mc_dropout


CUDA_VISIBLE_DEVICES=0 python test.py --net dgcnn --ckpt_path ckpts/graspqnet-g1b.tar --test_split test_seen --mc_dropout
CUDA_VISIBLE_DEVICES=1 python test.py --net dgcnn --ckpt_path ckpts/graspqnet-g1b.tar --test_split test_similar --mc_dropout
CUDA_VISIBLE_DEVICES=2 python test.py --net dgcnn --ckpt_path ckpts/graspqnet-g1b.tar --test_split test_novel --mc_dropout


CUDA_VISIBLE_DEVICES=0 python test.py --net dgcnn --ckpt_path ckpts/graspqnet-g1b.tar --test_split test_novel --mc_dropout

