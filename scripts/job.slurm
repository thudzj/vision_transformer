#!/bin/bash
#SBATCH --job-name=xlnet2    # create a short name for your job
#SBATCH --output=log_xlnet2.txt
#SBATCH --nodes=4                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks across all nodes
#SBATCH --cpus-per-task=32        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=128G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --hint=multithread
#SBATCH --time=202:00:00          # total run time limit (HH:MM:SS)

srun scripts/xlnet2.sh
