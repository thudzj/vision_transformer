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
pip install --upgrade jax==0.2.17 jaxlib==0.1.65+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install -r vit_jax/requirements.txt
pip install git+https://github.com/deepmind/jmp
to test: python3 -c "from jax.lib import xla_bridge; print(xla_bridge.get_backend().platform)"
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
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vit_jax.main --workdir=./mae --config=./vit_jax/configs/mae.py:b16 --config.dataset=/data/LargeData/Large/ImageNet/ --config.batch=512 --config.batch_eval=40
```

## finetune mae:  
```
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vit_jax.main --workdir=./ft_mae --config=./vit_jax/configs/ft.py:b16 --config.dataset=/data/LargeData/Large/ImageNet/ --config.pretrained_path=./mae --config.batch=256 --config.batch_eval=256
```

## pretrain xlnet:  
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m vit_jax.main --workdir=./xlnet --config=./vit_jax/configs/xlnet.py:b16 --config.dataset=/data/LargeData/Large/ImageNet/  --config.batch=768 --config.batch_eval=80

CUDA_VISIBLE_DEVICES=0,2,3,4,5,6 python -m vit_jax.main --workdir=./xlnet-mask195 --config=./vit_jax/configs/xlnet.py:b16 --config.dataset=/data/LargeData/Large/ImageNet/ --config.batch=576 --config.batch_eval=60 --config.num_mask=195
```

type2
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m vit_jax.main --workdir=./xlnet2 --config=./vit_jax/configs/xlnet.py:b16 --config.dataset=/data/LargeData/Large/ImageNet/ --config.encoder.predict_pos=True --config.out_dim=196 --config.sigma2=0.2 --config.batch=768 --config.batch_eval=80

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m vit_jax.main --workdir=./xlnet2-mask195 --config=./vit_jax/configs/xlnet.py:b16 --config.dataset=/data/LargeData/Large/ImageNet/ --config.encoder.predict_pos=True --config.out_dim=196 --config.sigma2=0.2 --config.batch=672 --config.batch_eval=70 --config.num_mask=195
```

## finetune xlnet(2):  
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m vit_jax.main --workdir=./ft_xlnet2 --config=./vit_jax/configs/ft.py:b16 --config.dataset=/data/LargeData/Large/ImageNet/ --config.pretrained_path=./xlnet2 --config.batch=512 --config.batch_eval=512

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m vit_jax.main --workdir=./ft_xlnet2-mask195 --config=./vit_jax/configs/ft.py:b16 --config.dataset=/data/LargeData/Large/ImageNet/ --config.pretrained_path=./xlnet2-mask195 --config.batch=448 --config.batch_eval=448
```


## run by slurm
to debug: 
```
srun -t 0:30:00 -N 1 --gres=gpu:4 --pty /bin/bash -l; source /apps/local/conda_init.sh; conda activate hao_vit
```

to run: `sbatch run.sh`