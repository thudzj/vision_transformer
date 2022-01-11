#!/bin/bash

WORLD_SIZE_JOB=$SLURM_NTASKS
RANK_NODE=$SLURM_NODEID
PROC_PER_NODE=4
MASTER_ADDR_JOB=$SLURM_SUBMIT_HOST
MASTER_PORT_JOB="12234"

module load nvidia/cuda/11.1

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

source /apps/local/conda_init.sh
conda activate hao_vit

### srun python3 vit_torch/run_pretraining.py --num_workers 10 --data_path /l/users/hongyiwa/datasets/ILSVRC2012/train --model pretrain_xlnet_base_patch16_224 --batch_size 256 --opt adamw --opt_betas 0.9 0.95 --warmup_epochs 40 --epochs 400 --output_dir /l/users/hongyiwa/hao/vision_transformer/logs/pretrain_xlnet2_base_patch16_224 --mask_ratio 0.995 --num_targets 49 --pred_pos --pred_pos_smoothing 0.2

python3 -m torch.distributed.launch \
	--nnodes=$WORLD_SIZE_JOB \
	--node_rank=$RANK_NODE \
	--nproc_per_node=$PROC_PER_NODE \
	--master_addr=$MASTER_ADDR_JOB \
	--master_port=$MASTER_PORT_JOB \
	vit_torch/run_pretraining.py --num_workers 1 \
	--data_path /l/users/hongyiwa/datasets/ILSVRC2012/train \
	--model pretrain_xlnet_base_patch16_224 \
	--batch_size 256 --opt adamw --opt_betas 0.9 0.95 --warmup_epochs 40 --epochs 400 \
	--output_dir /l/users/hongyiwa/hao/vision_transformer/logs/pretrain_xlnet2_base_patch16_224 \
	--mask_ratio 0.995 --num_targets 49 --pred_pos --pred_pos_smoothing 0.2
