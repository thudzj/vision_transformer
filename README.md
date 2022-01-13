## prepare the env
```
refer to https://gist.github.com/hungntt/836a3862dbe09dd643758ecbcbec043f for installing cuda
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh
eval "$(/home/ubuntu/anaconda3/bin/conda shell.bash hook)"
conda init
conda config --set auto_activate_base false
conda install cudatoolkit=11.1 cudnn -c pytorch -c conda-forge
sudo ln -s  /usr/local/cuda-11.1/lib64/libcupti.so.11.1 /usr/local/cuda-11.1/lib64/libcupti.so.11.0
# source prepare.sh
```

## to download imagenet
```
pip3 install kaggle  
vim ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
kaggle competitions download -c imagenet-object-localization-challenge
unzip imagenet-object-localization-challenge.zip
tar -xvzf imagenet_object_localization_patched2019.tar.gz
python3 in_val_process.py
```

## pretrain mae:  
```
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --master_port 60660 --nproc_per_node=4 vit_torch/run_pretraining.py --data_path /data/LargeData/Large/ImageNet/train --mask_ratio 0.75 --model pretrain_mae_base_patch16_224 --batch_size 128 --opt adamw --opt_betas 0.9 0.95 --warmup_epochs 40 --epochs 400 --output_dir logs/pretrain_mae_base_patch16_224
```

## finetune mae:  
```
python -m torch.distributed.launch --nproc_per_node=8 vit_torch/run_class_finetuning.py --model vit_base_patch16_224 --data_path /data/LargeData/Large/ImageNet/ --finetune logs/pretrain_mae_base_patch16_224/checkpoint.pth --output_dir logs/ft_mae_base_patch16_224 --batch_size 128 --opt adamw --opt_betas 0.9 0.999 --weight_decay 0.05 --epochs 100 --dist_eval
```


## pretrain xlnet:
if simple node, use the head `python -m torch.distributed.launch --master_port 60660 --nproc_per_node=8`  

```
NCCL_SOCKET_IFNAME=ib0 python -m torch.distributed.launch --nnodes=2 --node_rank=0 --nproc_per_node=8 --master_addr="11.4.3.28" --master_port=7788 vit_torch/run_pretraining.py --data_path /data/LargeData/Large/ImageNet/train --model pretrain_xlnet_base_patch16_224 --batch_size 128 --opt adamw --opt_betas 0.9 0.95 --warmup_epochs 40 --epochs 400 --output_dir logs/pretrain_xlnet_base_patch16_224 --mask_ratio 0.995 --num_targets 49

NCCL_SOCKET_IFNAME=ib0 python -m torch.distributed.launch --nnodes=2 --node_rank=1 --nproc_per_node=8 --master_addr="11.4.3.28" --master_port=7788 vit_torch/run_pretraining.py --data_path /data/LargeData/Large/ImageNet/train --model pretrain_xlnet_base_patch16_224 --batch_size 128 --opt adamw --opt_betas 0.9 0.95 --warmup_epochs 40 --epochs 400 --output_dir logs/pretrain_xlnet_base_patch16_224 --mask_ratio 0.995 --num_targets 49
```

ft
```
NCCL_SOCKET_IFNAME=ib0 python -m torch.distributed.launch --nnodes=2 --node_rank=0 --nproc_per_node=8 --master_addr="11.4.3.28" --master_port=7788 vit_torch/run_class_finetuning.py --model vit_base_patch16_224 --data_path /data/LargeData/Large/ImageNet/ --finetune logs/pretrain_xlnet_base_patch16_224/checkpoint-399.pth --output_dir logs/ft_xlnet_base_patch16_224 --batch_size 64 --opt adamw --opt_betas 0.9 0.999 --weight_decay 0.05 --epochs 100 --dist_eval

NCCL_SOCKET_IFNAME=ib0 python -m torch.distributed.launch --nnodes=2 --node_rank=1 --nproc_per_node=8 --master_addr="11.4.3.28" --master_port=7788 vit_torch/run_class_finetuning.py --model vit_base_patch16_224 --data_path /data/LargeData/Large/ImageNet/ --finetune logs/pretrain_xlnet_base_patch16_224/checkpoint-399.pth --output_dir logs/ft_xlnet_base_patch16_224 --batch_size 64 --opt adamw --opt_betas 0.9 0.999 --weight_decay 0.05 --epochs 100 --dist_eval
```

## pretrain xlnet2:
if simple node, use the head `python -m torch.distributed.launch --master_port 60660 --nproc_per_node=8`  

```
vit_torch/run_pretraining.py --data_path /home/ubuntu/ILSVRC/Data/CLS-LOC/train --model pretrain_xlnet_base_patch16_224 --batch_size 128 --opt adamw --opt_betas 0.9 0.95 --warmup_epochs 40 --epochs 400 --output_dir logs/pretrain_xlnet2_base_patch16_224 --mask_ratio 0.995 --num_targets 49 --pred_pos --pred_pos_smoothing 0.2
```

ft
```
vit_torch/run_class_finetuning.py --model vit_base_patch16_224 --data_path /home/ubuntu/ILSVRC/Data/CLS-LOC/ --finetune logs/pretrain_xlnet2_base_patch16_224/checkpoint-399.pth --output_dir logs/ft_xlnet2_base_patch16_224 --batch_size 64 --opt adamw --opt_betas 0.9 0.999 --weight_decay 0.05 --epochs 100 --dist_eval
```

## run by slurm
```
pip3 install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

to debug:
```
srun -t 0:30:00 -N 1 --gres=gpu:4 --pty /bin/bash -l; source /apps/local/conda_init.sh; conda activate hao_vit
```

to run: `sbatch run.sh`
