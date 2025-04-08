import equinox as eqx
from jax import random, Array


class Linear(eqx.Module):
    weight: Array
    bias: Array

    def __init__(self, in_size, out_size, key):
        wkey, bkey = random.split(key)
        self.weight = random.normal(wkey, (out_size, in_size))
        self.bias = random.normal(bkey, (out_size,))

    def __call__(self, x):
        return self.weight @ x + self.bias
