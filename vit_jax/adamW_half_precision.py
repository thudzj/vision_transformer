import flax
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


class Optimizer(flax.optim.OptimizerDef):
  """AdamW optimizer that may store state using half-precision."""

  @flax.struct.dataclass
  class HyperParams:
    learning_rate: np.ndarray
    beta1: np.ndarray
    beta2: np.ndarray
    eps: np.ndarray
    weight_decay: np.ndarray

  @flax.struct.dataclass
  class State:
    grad_ema: np.ndarray
    grad_sq_ema: np.ndarray

  def __init__(self,
               learning_rate=None,
               beta1=0.9,
               beta2=0.999,
               eps=1e-8,
               weight_decay=0.0,
               half_precision=False):
    hyper_params = Optimizer.HyperParams(learning_rate, beta1, beta2, eps, weight_decay)
    super().__init__(hyper_params)

    if half_precision:
      self.dtype = jnp.bfloat16 if jax.local_devices()[0].platform == 'tpu' else jnp.float16
    else:
      self.dtype = jnp.float32

  def init_param_state(self, param):
    return Optimizer.State(jnp.zeros_like(param, dtype=self.dtype), 
                           jnp.zeros_like(param, dtype=self.dtype))

  def apply_param_gradient(self, step, hyper_params, param, state, grad):
    assert hyper_params.learning_rate is not None, 'no learning rate provided.'
    beta1 = hyper_params.beta1
    beta2 = hyper_params.beta2
    weight_decay = hyper_params.weight_decay
    grad_sq = lax.square(grad)
    grad_ema = beta1 * state.grad_ema + (1. - beta1) * grad
    grad_sq_ema = beta2 * state.grad_sq_ema + (1. - beta2) * grad_sq

    # bias correction
    t = jnp.array(step + 1, lax.dtype(param.dtype))
    grad_ema_corr = grad_ema / (1 - beta1 ** t)
    grad_sq_ema_corr = grad_sq_ema / (1 - beta2 ** t)

    denom = jnp.sqrt(grad_sq_ema_corr) + hyper_params.eps
    new_param = param - hyper_params.learning_rate * grad_ema_corr / denom
    new_param -= hyper_params.learning_rate * weight_decay * param
    new_state = Optimizer.State(grad_ema.astype(self.dtype), grad_sq_ema.astype(self.dtype))
    return new_param, new_state

