## Before <code>pip install -r vit_jax/requirements.txt</code>, you should <code>pip install --upgrade jax==0.2.17 jaxlib==0.1.65+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html</code>


## pretrain mae:  
```
CUDA_VISIBLE_DEVICES=4,5,6 python -m vit_jax.main --workdir=./mae --config=./vit_jax/configs/mae.py:b16  --config.batch=192 --config.batch_eval=30
```

## finetune mae:  
```
CUDA_VISIBLE_DEVICES=4,5,6 python -m vit_jax.main --workdir=./ft_mae --config=./vit_jax/configs/ft_mae.py:b16  --config.batch=96 --config.batch_eval=96
```

## pretrain xlnet:  
```
CUDA_VISIBLE_DEVICES=4,5,6 python -m vit_jax.main --workdir=./xlnet --config=./vit_jax/configs/xlnet.py:b16  --config.batch=48 --config.batch_eval=30
```

## finetune xlnet:  
```
CUDA_VISIBLE_DEVICES=4,5,6 python -m vit_jax.main --workdir=./ft_xlnet --config=./vit_jax/configs/ft_xlnet.py:b16  --config.batch=96 --config.batch_eval=96
```
