## prepare the env
```
source prepare.sh
pip install --upgrade jax==0.2.17 jaxlib==0.1.65+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install -r vit_jax/requirements.txt
pip install git+https://github.com/deepmind/jmp
```

## pretrain mae:  
```
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vit_jax.main --workdir=./mae --config=./vit_jax/configs/mae.py:b16  --config.batch=512 --config.batch_eval=40
```

## finetune mae:  
```
CUDA_VISIBLE_DEVICES=4,5,6 python -m vit_jax.main --workdir=./ft_mae --config=./vit_jax/configs/ft_mae.py:b16  --config.batch=96 --config.batch_eval=96
```

## pretrain xlnet:  
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m vit_jax.main --workdir=./xlnet --config=./vit_jax/configs/xlnet.py:b16  --config.batch=896 --config.batch_eval=70
```

type2
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m vit_jax.main --workdir=./xlnet2 --config=./vit_jax/configs/xlnet2.py:b16  --config.batch=1024 --config.batch_eval=80
```

## finetune xlnet:  
```
CUDA_VISIBLE_DEVICES=4,5,6 python -m vit_jax.main --workdir=./ft_xlnet --config=./vit_jax/configs/ft_xlnet.py:b16  --config.batch=96 --config.batch_eval=96
```
