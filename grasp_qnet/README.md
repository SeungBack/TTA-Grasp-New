

conda activate testadagrasp 
tensorboard --logdir logs

# train



CUDA_VISIBLE_DEVICES=3 python main.py --net dgcnn --batch_size 64 --learning_rate 1e-4 --min_learning_rate 1e-6 --log_dir logs/final_new_do0.5_r0.25_bs64 --save_best_only --evals_per_epoch 8 --real_ratio=0.25

CUDA_VISIBLE_DEVICES=2 python main.py --net dgcnn --batch_size 32 --learning_rate 1e-4 --min_learning_rate 1e-6 --log_dir logs/final_new_do0.5_r0.25_bs32 --save_best_only --evals_per_epoch 8 --real_ratio=0.25


CUDA_VISIBLE_DEVICES=0,2 python main.py --net dgcnn --batch_size 256 --learning_rate 1e-4 --min_learning_rate 1e-6 --log_dir logs/final_new_do0.5_r0.5_bs256 --save_best_only --evals_per_epoch 8 --real_ratio=0.5

CUDA_VISIBLE_DEVICES=0 python main.py --net dgcnn --batch_size 128 --learning_rate 1e-4 --min_learning_rate 1e-6 --log_dir logs/final_new_do0.5_r0.5_bs128 --save_best_only --evals_per_epoch 8 --real_ratio=0.5


CUDA_VISIBLE_DEVICES=0,1 python main.py --net dgcnn --batch_size 256 --learning_rate 1e-4 --min_learning_rate 1e-6 --log_dir logs/final_new_do0.5_r0.5_bs256_bal_bal --save_best_only --evals_per_epoch 8 --real_ratio=0.5 --real_ratio=0.2

CUDA_VISIBLE_DEVICES=2 python main.py --net dgcnn --batch_size 128 --learning_rate 1e-4 --min_learning_rate 1e-6 --log_dir logs/final_new_do0.5_r0.5_bs128_bal_reg_bal --save_best_only --evals_per_epoch 8 --real_ratio=0.5 --real_ratio=0.2

CUDA_VISIBLE_DEVICES=3 python main.py --net dgcnn --batch_size 64 --learning_rate 1e-4 --min_learning_rate 1e-6 --log_dir logs/final_new_do0.5_r0.5_bs64_bal_reg_bal --save_best_only --evals_per_epoch 8 --real_ratio=0.5 --real_ratio=0.2


CUDA_VISIBLE_DEVICES=0 python main_reg.py --net dgcnn --batch_size 128 --learning_rate 1e-4 --min_learning_rate 1e-6 --log_dir logs/final_cross_att --save_best_only --evals_per_epoch 8 --real_ratio=0.5

CUDA_VISIBLE_DEVICES=0 python main_reg.py --net dgcnn --batch_size 128 --learning_rate 1e-4 --min_learning_rate 1e-6 --log_dir logs/final_basic_nodecay --save_best_only --evals_per_epoch 8 --real_ratio=0.5

CUDA_VISIBLE_DEVICES=1 python main_reg.py --net dgcnn --batch_size 128 --learning_rate 1e-4 --min_learning_rate 1e-6 --log_dir logs/final_basic_nodecay_do0.3 --save_best_only --evals_per_epoch 8 --real_ratio=0.5 --dropout 0.3

CUDA_VISIBLE_DEVICES=2 python main_reg.py --net dgcnn --batch_size 128 --learning_rate 1e-4 --min_learning_rate 1e-6 --log_dir logs/final_basic_nodecay_do0.1 --save_best_only --evals_per_epoch 8 --real_ratio=0.5 --dropout 0.1

CUDA_VISIBLE_DEVICES=1 python main_reg.py --net dgcnn --batch_size 128 --learning_rate 1e-4 --min_learning_rate 1e-6 --log_dir logs/final_basic_nodecay_do0.1_bn --save_best_only --evals_per_epoch 8 --real_ratio=0.5 --dropout 0.1


CUDA_VISIBLE_DEVICES=2 python main_reg.py --net dgcnn --batch_size 128 --learning_rate 1e-4 --min_learning_rate 1e-6 --log_dir logs/final_basic_nodecay_do0.1_bn --save_best_only --evals_per_epoch 2 --real_ratio=0.5 --dropout 0.1



CUDA_VISIBLE_DEVICES=0,1 python main_reg.py --net dgcnn --batch_size 256 --learning_rate 1e-4 --min_learning_rate 1e-6 --log_dir logs/final_basic_nodecay_do0.1_bn_bs256 --save_best_only --evals_per_epoch 2 --real_ratio=0.5 --dropout 0.1



CUDA_VISIBLE_DEVICES=0 python main_reg.py --net dgcnn --batch_size 128 --learning_rate 1e-4 --min_learning_rate 1e-6 --log_dir logs/final_basic_nodecay_do0.1_bn_real_only --save_best_only --evals_per_epoch 2 --real_ratio=0.5 --dropout 0.

CUDA_VISIBLE_DEVICES=0,1 python main_reg.py --net dgcnn --batch_size 256 --learning_rate 1e-4 --min_learning_rate 1e-6 --log_dir logs/final_basic_nodecay_do0.1_bn_bs256_0.0 --save_best_only --evals_per_epoch 2 --real_ratio=0.25 --dropout 0.1


CUDA_VISIBLE_DEVICES=0,1 python main_reg.py --net dgcnn --batch_size 256 --learning_rate 1e-4 --min_learning_rate 1e-6 --log_dir logs/final_final_basic_nodecay_do0.1_bn_bs256_0.25 --save_best_only --evals_per_epoch 2 --real_ratio=0.25 --dropout 0.1

CUDA_VISIBLE_DEVICES=0,1 python main_reg.py --net dgcnn --batch_size 256 --learning_rate 1e-4 --min_learning_rate 1e-6 --log_dir logs/final_final_basic_nodecay_do0.3_bn_bs256_0.25 --save_best_only --evals_per_epoch 2 --real_ratio=0.25 --dropout 0.3

CUDA_VISIBLE_DEVICES=0,1 python main_reg.py --net dgcnn --batch_size 256 --learning_rate 1e-4 --min_learning_rate 1e-6 --log_dir logs/real-final --save_best_only --evals_per_epoch 2 --real_ratio=0.25 --dropout 0.1

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

