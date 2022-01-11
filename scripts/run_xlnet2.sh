#!/bin/bash
#SBATCH --job-name=xlnet2    # create a short name for your job
#SBATCH --output=log_xlnet2.txt
#SBATCH --nodes=4                # node count
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

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export WORLD_SIZE=16

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

source /apps/local/conda_init.sh
conda activate hao_vit

srun python3 vit_torch/run_pretraining.py --num_workers 0 --data_path /l/users/hongyiwa/datasets/ILSVRC2012/train --model pretrain_xlnet_base_patch16_224 --batch_size 256 --opt adamw --opt_betas 0.9 0.95 --warmup_epochs 40 --epochs 400 --output_dir /l/users/hongyiwa/hao/vision_transformer/logs/pretrain_xlnet2_base_patch16_224 --mask_ratio 0.995 --num_targets 49 --pred_pos --pred_pos_smoothing 0.2
