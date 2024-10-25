import btc2sim
import parabellum as pb
import numpy as np
import jax.numpy as jnp
from chex import dataclass


def create_env(place):
    scenario = pb.env.scenario_fn(place, 100)
    env = pb.Environment(scenario)
    env_info = btc2sim.info.env_info_fn(env)
    # _, distances = compute_bfs(1 - (jnp.logical_or(env.terrain.building, env.terrain.water)), (45, 25))  # TODO: change
    # agents_info = btc2sim.info.agent_info_fn(env, {f"ally_{i}": distances for i in range(env.num_allies)})
    agents_info = btc2sim.info.agent_info_fn(env, {})
    return env, env_info, agents_info


def compute_bfs(mask, goal):
    """
    Start from goal on a grid world to compute the shortest path to the goal from any reachable cells
    """
    n = len(mask)
    neighbors = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # N, E, S, W like in Parabellum
    current_x = [goal[0]]
    current_y = [goal[1]]
    cost = 0
    directions = np.ones(mask.shape) * -1
    costs = np.ones(mask.shape) * (n**2)
    costs[goal] = cost
    while len(current_x) > 0:
        cost += 1
        new_x = np.empty(0, dtype=int)
        new_y = np.empty(0, dtype=int)

        for n_id, (i, j) in enumerate(neighbors):
            neighbors_x = np.array(current_x) + i
            neighbors_y = np.array(current_y) + j
            inside_mask = np.where(
                np.logical_and(
                    np.logical_and(neighbors_x >= 0, neighbors_x < n), np.logical_and(neighbors_y >= 0, neighbors_y < n)
                )
            )
            idxs = (neighbors_x[inside_mask], neighbors_y[inside_mask])
            valid = np.where(np.logical_and(costs[idxs] == n**2, mask[idxs]))
            valid_idx = inside_mask[0][valid[0]]
            idxs = (neighbors_x[valid_idx], neighbors_y[valid_idx])
            directions[idxs] = n_id
            costs[idxs] = cost

            new_x = np.concatenate([new_x, neighbors_x[valid_idx]])
            new_y = np.concatenate([new_y, neighbors_y[valid_idx]])

        current_x, current_y = new_x, new_y
    return np.array(directions, dtype=int), np.array(costs, dtype=int)
