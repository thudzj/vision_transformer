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


if simple node, use the head `python -m torch.distributed.launch --master_port 60660 --nproc_per_node=8`  

## mae:
```
src/main_pretrain.py --batch_size 64 --accum_iter 4 --model mae_vit_base_patch16 --norm_pix_loss --mask_ratio 0.75 --epochs 400 --warmup_epochs 40 --blr 1.5e-4
```

## xlnet:
```
NCCL_SOCKET_IFNAME=ib0 python -m torch.distributed.launch --nnodes=2 --node_rank=0 --nproc_per_node=8 --master_addr="11.4.3.28" --master_port=7788
NCCL_SOCKET_IFNAME=ib0 python -m torch.distributed.launch --nnodes=2 --node_rank=1 --nproc_per_node=8 --master_addr="11.4.3.28" --master_port=7788
src/main_pretrain.py --batch_size 128 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4 --mask_ratio 0.99 --num_targets 49

src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.034 Acc@5 96.302 loss 0.759

src/main_linprobe.py --data_path /data/LargeData/Large/ImageNet/ --model vit_base_patch16 --cls_token --epochs 90 --blr 0.1 --weight_decay 0.0 --dist_eval --finetune logs/pretrain_xlnet_base_patch16_224/checkpoint-399.pth --output_dir logs/linprob_xlnet_base_patch16_224 --batch_size 1024 --num_workers 16
    * Acc@1 51.088 Acc@5 74.608 loss 2.220
```

## xlnet2:
```
python3 -m torch.distributed.launch --nnodes=2 --node_rank=0 --nproc_per_node=8 --master_addr="172.31.38.116" --master_port=7788  
python3 -m torch.distributed.launch --nnodes=2 --node_rank=1 --nproc_per_node=8 --master_addr="172.31.38.116" --master_port=7788
src/main_pretrain.py --batch_size 256 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4 --mask_ratio 0.99 --num_targets 49 --pred_pos --pred_pos_smoothing 0.2 --tag type2

src/main_finetune.py --finetune logs/pretrain_xlnet2_base_patch16_224/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
```

## xlnet_m0.95:
```
python3 -m torch.distributed.launch --nnodes=2 --node_rank=0 --nproc_per_node=8 --master_addr="172.31.38.116" --master_port=7788 src/main_pretrain.py --batch_size 256 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4 --tag m0.95 --mask_ratio 0.95 --num_targets 49

src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.95/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.036 Acc@5 96.282 loss 0.759
```


## xlnet_m0.85
```
NCCL_SOCKET_IFNAME=ib0 python -m torch.distributed.launch --nnodes=3 --node_rank=0 --nproc_per_node=8 --master_addr="11.4.3.28" --master_port=7788 src/main_pretrain.py --batch_size 128 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4 --tag m0.85  --mask_ratio 0.85 --num_targets 49

src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.85/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    "train_loss": 2.668321636577638, "test_loss": 0.7567836930128661, "test_acc1": 83.10340692809356, "test_acc5": 96.36316376928328
```


## xlnet_gdepth13
```
python3 -m torch.distributed.launch --nnodes=3 --node_rank=0 --nproc_per_node=8 --master_addr="172.31.19.215" --master_port=7788 src/main_pretrain.py --batch_size 128 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4 --tag gdepth13 --mask_ratio 0.99 --num_targets 49 --g_depth 13

src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_gdepth13/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
```

## xlnet_gdepth7
```
NCCL_SOCKET_IFNAME=ib0 python -m torch.distributed.launch --nnodes=3 --node_rank=0 --nproc_per_node=8 --master_addr="11.4.3.28" --master_port=7788 src/main_pretrain.py --batch_size 128 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4 --tag gdepth7 --mask_ratio 0.99 --num_targets 49 --g_depth 7

src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_gdepth7/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 82.054 Acc@5 95.993 loss 0.792
```

## xlnet_gdepth1
```
python3 -m torch.distributed.launch --nnodes=2 --node_rank=0 --nproc_per_node=8 --master_addr="172.31.38.116" --master_port=7788 src/main_pretrain.py --batch_size 256 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4 --tag gdepth1 --mask_ratio 0.99 --num_targets 49 --g_depth 1

src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_gdepth1/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 82.456 Acc@5 96.020 loss 0.779
```

## num_targets 25
```
python3 -m torch.distributed.launch --nnodes=2 --node_rank=0 --nproc_per_node=8 --master_addr="172.31.38.116" --master_port=7788 src/main_pretrain.py --batch_size 256 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4 --tag tar25 --mask_ratio 0.99 --num_targets 25

src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_tar25/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 82.368 Acc@5 95.982 loss 0.793
```

## xlnet2 target 64
```
python -m torch.distributed.launch --master_port 60660 --nproc_per_node=8 src/main_pretrain.py  --batch_size 128 --accum_iter 4 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4 --tag type2_tar64 --mask_ratio 0.99 --num_targets 64 --pred_pos --pred_pos_smoothing 0.2
```


## xlnet span 1 2 4 7
```
src/main_pretrain.py --batch_size 256 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4 --mask_ratio 0.99 --num_targets 49 --span 1 2 4 7 --tag span1247
```

## xlnet target 81 epochs 800
```
python3 -m torch.distributed.launch --nnodes=4 --nproc_per_node=8 --master_addr="172.31.19.215" --master_port=7788 --node_rank=0  src/main_pretrain.py --batch_size 128 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 800 --warmup_epochs 40 --blr 1.5e-4 --output_dir logs/pretrain_xlnet_base_patch16_224_tar81 --mask_ratio 0.99 --num_targets 81

src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_tar81/checkpoint-399.pth --batch_size 32 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval

```
