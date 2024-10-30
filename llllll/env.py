import btc2sim
import parabellum as pb
import numpy as np
import jax.numpy as jnp
from chex import dataclass


def create_env(place):
    if place.strip() == "paradise":
        print(s_config["place"], pb.terrain_db.db.keys())
        print("———————————————————————————————————————————————————————")
        print("———————————————————————————————————————————————————————")
        print("———————————————————————————————————————————————————————")
        print("———————————————————————————————————————————————————————")
        print("———————————————————————————————————————————————————————")
        print("———————————————————————————————————————————————————————")
        scenario = pb.env.make_scenario(**s_config)
    else:
        scenario = pb.env.scenario_fn(place, 100)
    env = pb.Environment(scenario)
    env_info = btc2sim.info.env_info_fn(env)
    # _, distances = compute_bfs(1 - (jnp.logical_or(env.terrain.building, env.terrain.water)), (45, 25))  # TODO: change
    # agents_info = btc2sim.info.agent_info_fn(env, {f"ally_{i}": distances for i in range(env.num_allies)})
    agents_info = btc2sim.info.agent_info_fn_to_vmap(env, {})
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
    return jnp.array(directions, dtype=int), jnp.array(costs, dtype=int)


########################################################################################################################


n_allies = 10
n_enemies = 10
s_config = {
    "allies_type": [0] * 5 + [1] * 5,
    "n_allies": n_allies,
    "enemies_type": [0] * (n_enemies // 2) + [1] * (n_enemies // 2),
    "n_enemies": n_enemies,
    "size": 100,
    "place": "playground2",
    "unit_starting_sectors": [
        ([i for i in range(n_allies)], [0.25, 0.0, 0.5, 0.1]),
        ([i + n_allies for i in range(n_enemies // 2)], [0.65, 0.65, 0.2, 0.2]),
        ([i + n_allies for i in range(n_enemies // 2, n_enemies)], [0.25, 0.65, 0.2, 0.2]),
    ],
}

map_description = f"""
## Pertinent elements on the map:
West forest: trees at (0, 0) - (20, 100)
East River: water at (80, 0) - (90, 100)
East River's Bridge: normal at (80, 10) - (90, 20)
"""

winning_objective = {"elimination": None}
loosing_objective = {"elimination": None}

enemy_plan = f"""Step 0:
prerequisites: []
objective: elimination all
units: {[i for i in range(n_enemies//2)]}
- target position: {(s_config["size"]*3//4, int(s_config["size"]*3/4))}
- behavior: attack_in_close_range any
units: {[i for i in range(n_enemies//2, n_enemies)]}
- target position: {(s_config["size"]//4, int(s_config["size"]*3/4))}
- behavior: defend any
"""

winning_plan = f"""Step 0:
prerequisites: []
objective: elimination {[i for i in range(n_enemies//2)]}
units: {[5, 6, 7, 8, 9]}
- target position: {(int(s_config["size"]*0.9), int(s_config["size"]*0.75))}
- behavior: attack_static any

Step 1:
prerequisites: []
objective: position
units: {[i for i in range(5)]}
- target position: {(int(s_config["size"]*0.1), int(s_config["size"]*0.1))}
- behavior: attack_static any

Step 2:
prerequisites: [1]
objective: position
units: {[i for i in range(5)]}
- target position: {(int(s_config["size"]*0.1), int(s_config["size"]*0.75))}
- behavior: attack_static any

Step 3:
prerequisites: [2]
objective: elimination {[i for i in range(n_enemies//2, n_enemies)]}
units: {[i for i in range(5)]}
- target position: {(int(s_config["size"]*0.25), int(s_config["size"]*0.75))}
- behavior: attack_in_close_range any
"""

whatever_you_want = {
    "config": s_config,
    "enemy_plan": enemy_plan,
    "ally_plan": winning_plan,
    # "default_plan": default_plan,
    # "introduction": elimination_introduction,
    "map_description": map_description,
    "winning_objective": winning_objective,
    "loosing_objective": loosing_objective,
}
