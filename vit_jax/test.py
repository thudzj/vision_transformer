import numpy as np
import math
import jax.numpy as jnp
import contextlib

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally: 
        np.set_printoptions(**original)

num_patches = 196
sigma2 = 0.2

masks = jnp.asarray(np.stack([np.random.permutation(196)[:10], np.random.permutation(196)[:10]], 0))
print(masks)

ncol = int(math.sqrt(num_patches))
row_labels = masks.reshape(-1, 1, 1) // ncol
col_labels = masks.reshape(-1, 1, 1) % ncol
labels = jnp.exp(-((row_labels - jnp.arange(ncol).reshape(1, -1, 1)) ** 2 + 
        (col_labels - jnp.arange(ncol).reshape(1, 1, -1)) ** 2) / 2. / sigma2)
labels = labels.reshape(2, -1, num_patches)
labels = labels / jnp.sum(labels, axis=-1, keepdims=True)

print(labels.shape)

with printoptions(precision=3, suppress=True, linewidth=100000,):
    print(labels[0, 1].reshape(14, 14))

    print(labels[1, 2].reshape(14, 14))