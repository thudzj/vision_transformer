#!/bin/bash
#SBATCH --job-name=xlnet2_1node    # create a short name for your job
#SBATCH --output=log_xlnet2_1node.txt
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=4      # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=128G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --hint=multithread
#SBATCH --time=202:00:00          # total run time limit (HH:MM:SS)

module load nvidia/cuda/11.1

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

source /apps/local/conda_init.sh
conda activate hao_vit

python3 -m torch.distributed.launch \
    --nproc_per_node=4 \
    vit_torch/run_pretraining.py --num_workers 0 \
    --data_path /l/users/hongyiwa/datasets/ILSVRC2012/train \
    --model pretrain_xlnet_base_patch16_224 \
    --batch_size 256 --opt adamw --opt_betas 0.9 0.95 --warmup_epochs 40 --epochs 400 \
    --output_dir /l/users/hongyiwa/hao/vision_transformer/logs/pretrain_xlnet2_base_patch16_224 \
    --mask_ratio 0.995 --num_targets 49 --pred_pos --pred_pos_smoothing 0.2
