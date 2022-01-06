#!/bin/bash
#SBATCH --job-name=xlnet2-mask195    # create a short name for your job
#SBATCH --output=log_xlnet2-mask195.txt
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks across all nodes
#SBATCH --cpus-per-task=24        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --time=202:00:00          # total run time limit (HH:MM:SS)

module load nvidia/cuda/11.4

source /apps/local/conda_init.sh
conda activate hao_vit
rm -rf /l/users/hongyiwa/hao/vision_transformer/xlnet2-mask195

XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda python3 -m vit_jax.main --workdir=/l/users/hongyiwa/hao/vision_transformer/xlnet2-mask195 --config=/l/users/hongyiwa/hao/vision_transformer/vit_jax/configs/xlnet.py:b16 --config.dataset=/l/users/hongyiwa/datasets/ILSVRC2012 --config.encoder.predict_pos=True --config.out_dim=196 --config.sigma2=0.2 --config.batch=1024 --config.batch_eval=40 --config.num_mask=195 --config.flip=False
