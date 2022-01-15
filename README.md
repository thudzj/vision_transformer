## prepare the env
```
refer to https://gist.github.com/hungntt/836a3862dbe09dd643758ecbcbec043f for installing cuda
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh
eval "$(/home/ubuntu/anaconda3/bin/conda shell.bash hook)"
conda init
conda config --set auto_activate_base false
conda install cudatoolkit=11.1 cudnn -c pytorch -c conda-forge
pip3 install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# sudo ln -s  /usr/local/cuda-11.1/lib64/libcupti.so.11.1 /usr/local/cuda-11.1/lib64/libcupti.so.11.0
# source prepare.sh
pip3 install timm==0.3.2
```

edit `python3.6/site-packages/timm/models/layers/helpers.py`
```
import torch
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs
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


## pretrain xlnet:
if simple node, use the head `python -m torch.distributed.launch --master_port 60660 --nproc_per_node=8`  

```
NCCL_SOCKET_IFNAME=ib0 python -m torch.distributed.launch --nnodes=2 --node_rank=0 --nproc_per_node=8 --master_addr="11.4.3.28" --master_port=7788
NCCL_SOCKET_IFNAME=ib0 python -m torch.distributed.launch --nnodes=2 --node_rank=1 --nproc_per_node=8 --master_addr="11.4.3.28" --master_port=7788
src/main_pretrain.py --data_path /data/LargeData/Large/ImageNet/ --batch_size 128 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4 --weight_decay 0.05 --output_dir logs/pretrain_xlnet_base_patch16_224 --accum_iter 2 --mask_ratio 0.99 --num_targets 49
```

ft-bs1024 (n16 * b64)
```
https://github.com/facebookresearch/mae/blob/main/FINETUNE.md
src/main_finetune.py --data_path /data/LargeData/Large/ImageNet/ --finetune logs/pretrain_xlnet_base_patch16_224/checkpoint-399.pth --output_dir logs/ft_xlnet_base_patch16_224 --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.65 --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
```

## pretrain xlnet2:
if simple node, use the head `python3 -m torch.distributed.launch --master_port 60660 --nproc_per_node=8`  

```
python3 -m torch.distributed.launch --nnodes=2 --node_rank=0 --nproc_per_node=8 --master_addr="172.31.38.116" --master_port=7788  
python3 -m torch.distributed.launch --nnodes=2 --node_rank=1 --nproc_per_node=8 --master_addr="172.31.38.116" --master_port=7788
src/main_pretrain.py --data_path /home/ubuntu/ILSVRC/Data/CLS-LOC/ --batch_size 256 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4 --weight_decay 0.05 --output_dir logs/pretrain_xlnet2_base_patch16_224 --mask_ratio 0.99 --num_targets 49 --pred_pos --pred_pos_smoothing 0.2
```

ft-bs1024 (n16 * b64)
```
python3 -m torch.distributed.launch --nnodes=2 --node_rank=0 --nproc_per_node=8 --master_addr="172.31.22.198" --master_port=7788
python3 -m torch.distributed.launch --nnodes=2 --node_rank=1 --nproc_per_node=8 --master_addr="172.31.22.198" --master_port=7788
src/main_finetune.py --data_path /home/ubuntu/zhijie/ILSVRC/Data/CLS-LOC/ --finetune logs/pretrain_xlnet2_base_patch16_224/checkpoint-399.pth --output_dir logs/ft_xlnet2_base_patch16_224 --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.65 --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
```

## pretrain xlnet_m0.95:
```
python3 -m torch.distributed.launch --nnodes=2 --node_rank=0 --nproc_per_node=8 --master_addr="172.31.22.198" --master_port=7788
python3 -m torch.distributed.launch --nnodes=2 --node_rank=1 --nproc_per_node=8 --master_addr="172.31.22.198" --master_port=7788
src/main_pretrain.py --data_path /home/ubuntu/zhijie/ILSVRC/Data/CLS-LOC/ --batch_size 256 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4 --weight_decay 0.05 --output_dir logs/pretrain_xlnet_base_patch16_224_m0.95 --mask_ratio 0.95 --num_targets 49
```
