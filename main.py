# %% Imports
import equinox as eqx
import parabellum as pb
from jax import random, lax, vmap
import jax.numpy as jnp
import numpy as np
from PIL import Image
from omegaconf import OmegaConf


# %% Setup #################################################################
n_steps = 100
cfg = OmegaConf.load("conf.yaml")
rng, key = random.split(random.PRNGKey(0))
env, scene = pb.env.Env(cfg=cfg), pb.env.scene_fn(cfg)


# %% Model


# %% Constants
env = pb.env.Env(cfg=(cfg := pb.env.Conf()))
batch_size, hidden_size, in_size, out_size = 32, 4, 2, 3
x = jnp.ones((batch_size, in_size))
y = jnp.zeros((batch_size, out_size))
model = Model(in_size, out_size, 10, 3, key=random.PRNGKey(0))


@eqx.filter_grad
@eqx.filter_jit
def loss_fn(model, x, y):
    pred = vmap(model)(x)
    return jnp.mean((pred - y) ** 2)


grads = loss_fn(model, x, y)
print(grads)

exit()


# %% Functions
def step(carry, rng):
    obs, state = carry
    action_key, step_key = random.split(rng, (2, env.scene.num_agents))
    action = action_fn(env, action_key, state, obs, behavior, jnp.arange(env.scene.num_agents))
    obs, state = env.step(step_key, state, action)
    return state, (obs, state)


def anim(seq, scale=8, width=10):  # animate positions
    idxs = jnp.concat((jnp.arange(seq.shape[0]).repeat(seq.shape[1])[..., None], seq.reshape(-1, 2)), axis=1).T
    imgs = np.array(jnp.zeros((seq.shape[0], width, width)).at[*idxs].set(255)).astype(np.uint8)  # setting color
    imgs = [Image.fromarray(img).resize(np.array(img.shape[:2]) * scale, Image.NEAREST) for img in imgs]  # type: ignore
    imgs[0].save("output.gif", save_all=True, append_images=imgs[1:], duration=100, loop=0)


rng, key = random.split(random.PRNGKey(0))
env = pb.env.Env(cfg=(cfg := pb.env.Conf()))
batch_size, in_size, out_size = 32, 2, 3
model = Linear(in_size, out_size, key=key)
rngs = random.split(rng, 100)
obs, state = env.reset(key)
state, seq = lax.scan(step, (obs, state), rngs)
# anim(seq.unit_position.astype(int), width=env.cfg.size, scale=8)
