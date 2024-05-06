import optax
import jax.numpy as jnp
from jax import scipy as jscipy
from jax import random as jrand

class Optimizer:
    def __init__(self, optimizer, params):
        self.optimizer = optimizer
        self.params = params
        self.opt_state = optimizer.init(self.params)

    def apply_gradients(self, grads):
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)