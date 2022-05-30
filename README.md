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

src/main_finetune.py --finetune logs/pretrain_mae_base_patch16_224/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.65 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.133 Acc@5 96.451 loss 0.754
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

## xlnet span 1 2 4 7
```
src/main_pretrain.py --batch_size 256 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4 --mask_ratio 0.99 --num_targets 49 --span 1 2 4 7 --tag span1247

python3 -m torch.distributed.launch --nnodes=2 --nproc_per_node=8 --master_addr="172.31.38.116" --master_port=7788 --node_rank=0 src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_span1247/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 82.828 Acc@5 96.254 loss 0.772
```

## xlnet target 81 epochs 800
```
python3 -m torch.distributed.launch --nnodes=4 --nproc_per_node=8 --master_addr="172.31.19.215" --master_port=7788 --node_rank=0  src/main_pretrain.py --batch_size 128 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 800 --warmup_epochs 40 --blr 1.5e-4 --tag tar81 --mask_ratio 0.99 --num_targets 81

python -m torch.distributed.launch --master_port 60660 --nproc_per_node=8 src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_tar81/checkpoint-799.pth --batch_size 64 --accum_iter 2 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.65 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 82.856 Acc@5 96.272 loss 0.763

python -m torch.distributed.launch --master_port 60660 --nproc_per_node=8 src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_tar81/checkpoint-799.pth --batch_size 64 --accum_iter 2 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.050 Acc@5 96.298 loss 0.762
    Max accuracy: 83.15%

python -m torch.distributed.launch --master_port 60660 --nproc_per_node=8 src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_tar81/checkpoint-799.pth --batch_size 64 --accum_iter 2 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.85 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 82.832 Acc@5 96.214 loss 0.771

python -m torch.distributed.launch --master_port 60660 --nproc_per_node=8 src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_tar81/checkpoint-799.pth --batch_size 64 --accum_iter 2 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 1 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 82.588 Acc@5 96.124 loss 0.778


src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_tar81/checkpoint-799.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval --output_dir logs/ft_xlnet_base_patch16_224_tar81_avgclf_alpha1 --ar --alpha 1
    * Acc@1 81.640 Acc@5 95.773 loss 0.821  

src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_tar81/checkpoint-799.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval --output_dir logs/ft_xlnet_base_patch16_224_tar81_avgclf_alpha0 --ar --alpha 0
    * Acc@1 81.652 Acc@5 95.724 loss 0.827

src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_tar81/checkpoint-799.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval --output_dir logs/ft_xlnet_base_patch16_224_tar81_clsclf_alpha0 --ar --alpha 0 --cls_token
    * Acc@1 81.546 Acc@5 95.762 loss 0.822

src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_tar81/checkpoint-799.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval --output_dir logs/ft_xlnet_base_patch16_224_tar81_clsclf --cls_token
    * Acc@1 83.077 Acc@5 96.229 loss 0.769
```

--------------------------------------------------------------------------------
(new version -- class token not involved in the feature extraction):

## xlnet
```
src/main_pretrain.py --batch_size 256 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4 --mask_ratio 0.99 --num_targets 49 --one_extra_layer --tag oel

src/main_finetune.py --finetune newlogs/pretrain_xlnet_base_patch16_224_oel/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 82.620 Acc@5 96.098 loss 0.782
```

## xlnet2:
```
NCCL_SOCKET_IFNAME=ib0 python -m torch.distributed.launch --nnodes=3 --node_rank=0 --nproc_per_node=8 --master_addr="11.4.3.28" --master_port=7788 src/main_pretrain.py  --batch_size 128 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4 --mask_ratio 0.99 --num_targets 49 --pred_pos --pred_pos_smoothing 0.2 --tag type2 --one_extra_layer
```

## xlnet
```
src/main_pretrain.py --batch_size 128 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4 --mask_ratio 0.85 --num_targets 49 --one_extra_layer --tag oel_m0.85


src/main_finetune.py --finetune newlogs/pretrain_xlnet_base_patch16_224_oel_m0.85/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval --output_dir newlogs/ft_xlnet_base_patch16_224_oel_m0.85
    * Acc@1 82.996 Acc@5 96.300 loss 0.759

src/main_finetune.py --finetune newlogs/pretrain_xlnet_base_patch16_224_oel_m0.85/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval --output_dir newlogs/ft_xlnet_base_patch16_224_oel_m0.85_clsclf --cls_token
    * Acc@1 82.654 Acc@5 96.184 loss 0.773
```



------------------ return to the previous version -------------------

```
src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio_range 0.5 0.99 --num_targets 49 --tag m0.5-0.99
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.5-0.99/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.133 Acc@5 96.271 loss 0.766


src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio_range 0.5 0.99 --num_targets 49 --tag m0.5-0.99-structuredctx --structured_ctx
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.5-0.99-structuredctx/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.141 Acc@5 96.387 loss 0.754
    [11:08:00.131382] Accuracy of the network on the 50000 test images: 83.1%
    [11:08:00.131415] Max accuracy: 83.18%


NCCL_SOCKET_IFNAME=ib0 python -m torch.distributed.launch --nnodes=2 --node_rank=0 --nproc_per_node=8 --master_addr="11.4.3.28" --master_port=7788 src/main_pretrain.py --batch_size 64 --accum_iter 4 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio_range 0.5 0.99 --num_targets 49 --tag m0.5-0.99_avgmask --avg_mask_token
NCCL_SOCKET_IFNAME=ib0 python -m torch.distributed.launch --nnodes=2 --node_rank=0 --nproc_per_node=8 --master_addr="11.4.3.28" --master_port=7788 src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.5-0.99_avgmask/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 82.802 Acc@5 96.220 loss 0.776  
```


```
src/main_pretrain.py --batch_size 128 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.87 --num_targets 49 --tag m0.87-structuredctx --structured_ctx
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.87-structuredctx/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.119 Acc@5 96.311 loss 0.754


src/main_pretrain.py --batch_size 128 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-structuredctx --structured_ctx
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.335 Acc@5 96.401 loss 0.754


src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-structuredctx-2 --structured_ctx
src/main_pretrain.py --batch_size 128 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-structuredctx-2 --structured_ctx --resume logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx2/checkpoint-90.pth
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-2/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    "test_loss": 0.7517214033566415, "test_acc1": 83.17138518504584, "test_acc5": 96.41714650243807
    * Acc@1 83.237 Acc@5 96.427 loss 0.751



src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-structuredctx-nonorm --structured_ctx
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-nonorm/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 82.863 Acc@5 96.117 loss 0.775

src/main_pretrain.py --batch_size 128 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 800 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-structuredctx-800 --structured_ctx
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-800/checkpoint-799.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.269 Acc@5 96.511 loss 0.751


src/main_pretrain.py --batch_size 128 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 800 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.87 --num_targets 49 --tag m0.87-structuredctx-800 --structured_ctx
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.87-structuredctx-800/checkpoint-799.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.075 Acc@5 96.313 loss 0.763


src/main_pretrain.py --batch_size 128 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 800 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-structuredctx-800-da_v0r --structured_ctx --da aa-v0r
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-800-da_v0r/checkpoint-799.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.069 Acc@5 96.385 loss 0.752


src/main_pretrain.py --batch_size 128 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 800 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-structuredctx-800-scale.5 --structured_ctx --scale 0.5 1.0
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-800-scale.5/checkpoint-799.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.203 Acc@5 96.373 loss 0.755


src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 800 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-structuredctx-800-scale.8 --structured_ctx --scale 0.8 1.0
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-800-scale.8/checkpoint-799.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval


src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 800 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-800
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-800/checkpoint-799.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 82.999 Acc@5 96.347 loss 0.767



src/main_pretrain.py --batch_size 128 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.67 --num_targets 49 --tag m0.67-structuredctx --structured_ctx
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.67-structuredctx/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.141 Acc@5 96.411 loss 0.757


src/main_pretrain.py --batch_size 128 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.815 --num_targets 49 --tag m0.815-structuredctx --structured_ctx
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.815-structuredctx/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.131 Acc@5 96.397 loss 0.754


src/main_pretrain.py --batch_size 128 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-structuredctx-da_manual --structured_ctx --da manual
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-da_manual/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    "test_loss": 0.7578091232478619, "test_acc1": 83.09141077044981, "test_acc5": 96.36916185401604


src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-structuredctx-da_orir --structured_ctx --da aa-originalr
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-da_orir/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    "test_loss": 0.7554487223550678, "test_acc1": 83.08341333169015, "test_acc5": 96.30718169224545


src/main_pretrain.py --batch_size 128 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-structuredctx-da_v0r --structured_ctx --da aa-v0r
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-da_v0r/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.225 Acc@5 96.377 loss 0.759


src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-structuredctx-da_m9 --structured_ctx --da aa-rand-m9-mstd0.5-inc1
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-da_m9/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.095 Acc@5 96.317 loss 0.759




src/main_pretrain.py --batch_size 128 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-structuredctx-da_pa --structured_ctx --da patch_aug (CJ 16)
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-da_pa/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    have not fted

src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-structuredctx-da_pa4 --structured_ctx --da patch_aug
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-da_pa4/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    "test_loss": 0.7612201334536075, "test_acc1": 82.92746323922927, "test_acc5": 96.25719769139818



src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 81 --tag m0.75-structuredctx-tar81 --structured_ctx
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-tar81/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.179 Acc@5 96.297 loss 0.766


src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 64 --tag m0.75-structuredctx-tar64 --structured_ctx
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-tar64/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.133 Acc@5 96.343 loss 0.755

----------------

src/main_pretrain.py --batch_size 128 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.87 --num_targets 49 --tag m0.87
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.87/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.121 Acc@5 96.263 loss 0.756

src/main_pretrain.py --batch_size 128 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.163 Acc@5 96.275 loss 0.757


------------------

src/main_pretrain.py --batch_size 128 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.87 --num_targets 49 --tag m0.87-beitctx --beit_ctx
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.87-beitctx/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.187 Acc@5 96.353 loss 0.755

src/main_pretrain.py --batch_size 128 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-beitctx --beit_ctx
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-beitctx/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.145 Acc@5 96.263 loss 0.758

src/main_pretrain.py --batch_size 128 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 800 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.87 --num_targets 49 --tag m0.87-beit_ctx-800 --beit_ctx
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.87-beit_ctx-800/checkpoint-799.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.037 Acc@5 96.367 loss 0.760


src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 800 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-beit_ctx-800 --beit_ctx
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-beit_ctx-800/checkpoint-799.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.369 Acc@5 96.455 loss 0.748
    [14:23:07.176209] Accuracy of the network on the 50000 test images: 83.4%
    [14:23:07.176254] Max accuracy: 83.40%


src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 800 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-beit_ctx-800-scale.5 --beit_ctx --scale 0.5 1.0
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-beit_ctx-800-scale.5/checkpoint-799.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.359 Acc@5 96.533 loss 0.751


src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 800 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-beit_ctx-800-scale.8 --beit_ctx --scale 0.8 1.0
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-beit_ctx-800-scale.8/checkpoint-799.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.341 Acc@5 96.423 loss 0.750


src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 800 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-beit_ctx-800-scale.08 --beit_ctx --scale 0.08 1.0
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-beit_ctx-800-scale.08/checkpoint-799.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    "test_loss": 0.7501852255314588, "test_acc1": 83.2333653504011, "test_acc5": 96.37116122062741


src/main_pretrain.py --batch_size 256 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio_range 0.67 0.87 --num_targets 49 --tag m0.67-0.87-beit_ctx-400 --beit_ctx
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.67-0.87-beit_ctx-400/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.013 Acc@5 96.385 loss 0.756


src/main_pretrain.py --batch_size 256 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 800 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-all_beit_ctx-800 --all_beit_ctx
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-all_beit_ctx-800/checkpoint-799.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 82.951 Acc@5 96.265 loss 0.764




src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 800 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-beit_ctx-800-da_v0r --beit_ctx --da aa-v0r
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-beit_ctx-800-da_v0r/checkpoint-799.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.147 Acc@5 96.435 loss 0.757


src/main_pretrain.py --batch_size 256 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 1600 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-beit_ctx-1600 --beit_ctx
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-beit_ctx-1600/checkpoint-1599.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.247 Acc@5 96.279 loss 0.752



src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 800 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-beit_ctx-800-oel --beit_ctx --one_extra_layer
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-beit_ctx-800-oel/checkpoint-799.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.353 Acc@5 96.423 loss 0.748   


src/main_pretrain.py --batch_size 256 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 800 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-beit_ctx-800-oel --beit_ctx --one_extra_layer
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-beit_ctx-800-oel/checkpoint-799.pth --batch_size 128 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    has_proj = True
    * Acc@1 83.172 Acc@5 96.406 loss 0.759

--------------------------------------------------------

src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-structuredctx-400-oel --structured_ctx --one_extra_layer --resume logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-400-oel/checkpoint-290.pth
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-400-oel/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    has_proj = True
    * Acc@1 83.319 Acc@5 96.317 loss 0.751
    layer_decay 0.65 * Acc@1 82.977 Acc@5 96.315 loss 0.758
    layer_decay 0.65 blr 1e-3  * Acc@1 83.061 Acc@5 96.447 loss 0.752
    batch_size 32  * Acc@1 83.165 Acc@5 96.331 loss 0.757


src/main_pretrain.py --batch_size 64 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 2e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-structuredctx-blr2 --structured_ctx --scale 0.5 1.0
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-blr2/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.215 Acc@5 96.389 loss 0.755


src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-structuredctx-oel-gd6 --structured_ctx --g_depth 6 --one_extra_layer
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-oel-gd6/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    83.04


src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 2e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-structuredctx-newsetting --structured_ctx --one_extra_layer --betas 0.9 0.999 --clip_grad 0.02
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-newsetting/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 82.792 Acc@5 96.287 loss 0.773


src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.5 --num_targets 49 --tag m0.5-structuredctx-s0.5 --structured_ctx --scale 0.5 1.0
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.5-structuredctx-s0.5/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    83.3

src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.4 --num_targets 49 --tag m0.4-beit_ctx-s0.5 --beit_ctx --scale 0.5 1.0
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.4-beit_ctx-s0.5/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval



src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-structuredctx-s0.5 --structured_ctx --scale 0.5 1.0
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-s0.5/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    83.21


src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-structuredctx-s0.2 --structured_ctx --scale 0.2 1.0
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-s0.2/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.219 Acc@5 96.357 loss 0.758

src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-structuredctx-400-oel2 --structured_ctx --one_extra_layer
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-400-oel2/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval

src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-beit_ctx-s0.5 --beit_ctx --scale 0.5 1.0
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-beit_ctx-s0.5/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval

src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-beit_ctx-s0.8 --beit_ctx --scale 0.8 1.0
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-beit_ctx-s0.8/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval


src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_large_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-structuredctx-oel --structured_ctx --one_extra_layer
src/main_finetune.py --finetune logs/pretrain_xlnet_large_patch16_224_m0.75-structuredctx-oel/checkpoint-399.pth --batch_size 32 --model vit_large_patch16 --epochs 50 --blr 1e-3 --layer_decay 0.75 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval






-----------------------------
src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-structuredctx-400-oel2 --structured_ctx --one_extra_layer --betas 0.9 0.999
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-400-oel2/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 82.750 Acc@5 96.225 loss 0.769


src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-structuredctx-400-oel-s0.5 --structured_ctx --one_extra_layer --scale 0.5 1.0
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-400-oel-s0.5/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval


src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 1600 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-beit_ctx-1600 --beit_ctx
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-beit_ctx-1600/checkpoint-1599.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
  83.35
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-beit_ctx-1600/checkpoint-1599.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.65 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval --output_dir logs/ft_xlnet_base_patch16_224_m0.75-beit_ctx-1600-1
  * Acc@1 83.459 Acc@5 96.527 loss 0.739
  Max accuracy: 83.48%

```


## The final results
```
src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 1600 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-structuredctx-1600-oel --structured_ctx --one_extra_layer
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-1600-oel/checkpoint-1599.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.223 Acc@5 96.425 loss 0.750
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-1600-oel/checkpoint-1599.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.65 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval --output_dir logs/ft_xlnet_base_patch16_224_m0.75-structuredctx-1600-oel-1
    83.62
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-1600-oel/checkpoint-1599.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.65 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval --output_dir logs/ft_xlnet_base_patch16_224_m0.75-structuredctx-1600-oel-2
    * Acc@1 83.623 Acc@5 96.551 loss 0.733
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-1600-oel/checkpoint-1599.pth --batch_size 32 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.65 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval --output_dir logs/ft_xlnet_base_patch16_224_m0.75-structuredctx-1600-oel-3
    83.42


src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 400 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-structuredctx-400-oel --structured_ctx --one_extra_layer
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-400-oel/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.75 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    83.32
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-400-oel/checkpoint-399.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.65 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval --output_dir logs/ft_xlnet_base_patch16_224_m0.75-structuredctx-400-oel-2
    * Acc@1 82.977 Acc@5 96.315 loss 0.758

src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --norm_pix_loss --epochs 1600 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-structuredctx-1600 --structured_ctx
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-1600/checkpoint-1599.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.65 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.443 Acc@5 96.549 loss 0.740
      Max accuracy: 83.49%

src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --epochs 1600 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-structuredctx-1600-oel-nonorm --structured_ctx --one_extra_layer
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-1600-oel-nonorm/checkpoint-1599.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.65 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
    * Acc@1 83.015 Acc@5 96.403 loss 0.753
    [15:06:59.960647] Accuracy of the network on the 50000 test images: 83.0%
    [15:06:59.960688] Max accuracy: 83.07%

    CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --epochs 1600 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-structuredctx-1600-oel-nonorm --structured_ctx --one_extra_layer --resume logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-1600-oel-nonorm/checkpoint-1599.pth --generate

src/main_pretrain.py --batch_size 64 --accum_iter 2 --model xlnet_vit_base_patch16 --epochs 1600 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-structuredctx-1600-oel-nonorm-s0.875 --structured_ctx --one_extra_layer --scale 0.875 1
src/main_finetune.py --finetune logs/pretrain_xlnet_base_patch16_224_m0.75-structuredctx-1600-oel-nonorm-s0.875/checkpoint-1599.pth --batch_size 64 --model vit_base_patch16 --epochs 100 --blr 5e-4 --layer_decay 0.65 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval

src/main_pretrain.py --batch_size 16 --accum_iter 8 --model xlnet_vit_large_patch16 --norm_pix_loss --epochs 1600 --warmup_epochs 40 --blr 1.5e-4  --mask_ratio 0.75 --num_targets 49 --tag m0.75-structuredctx-1600-oel --structured_ctx --one_extra_layer
src/main_finetune.py --finetune logs/pretrain_xlnet_large_patch16_224_m0.75-structuredctx-1600-oel/checkpoint-1599.pth --batch_size 32 --model vit_large_patch16 --epochs 50 --blr 1e-3 --layer_decay 0.75 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval
```
