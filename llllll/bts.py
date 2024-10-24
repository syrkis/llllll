# # Imports

from btc2sim import btc2sim
from itertools import chain, combinations
from functools import lru_cache


# # Functions


# ## Target variants


# +
def powerset(iterable):
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))


unit_types = ["soldier", "sniper", "swat", "turret", "civilian"]


def choose_targets(bt_txt, targets):
    for t in targets:
        assert t in unit_types
    targets_txt = "any" if len(targets) == 0 else " or ".join([t for t in targets])
    return bt_txt.replace("any", targets_txt)


def compute_all_variants(bt_txt, unit_types):
    if "any" in bt_txt:
        subsets = [set(types) for types in powerset(unit_types) if types]
        variants = [choose_targets(bt_txt, subset) for subset in subsets]
        return subsets, variants
    else:
        return [None], [bt_txt]


def find_bt_variant(x, subsets, variants):
    assert x in subsets
    return variants[subsets.index(x)]


# -

# ## Follow map as default


def set_default(bt_txt, default):
    return f"F ( S( C (in_sight foe any) :: {bt_txt}) :: {default})"


# # BTs

bt_wait = "A (stand)"

bt_flee = "F ( S (C (in_sight foe any) :: A (move away_from closest foe any)) :: A (stand))"

# +
bt_attack_and_wait = """
    F (
        A (attack weakest any) ::
        A (stand)
    )
    """

bt_attack_and_chase = (
    "F(A (attack closest any) :: S (C (in_sight foe any) :: A (move toward closest foe any)) :: A (stand))"
)

bt_attack_and_stay_out_of_range = """
        F(
            A (attack weakest any) ::
            S (C (in_reach foe any) :: A (move away_from closest foe any)) ::
            S (C (in_sight foe any) :: A (move toward closest foe any)) ::
            A (stand)
        )
"""

bt_attack_and_fallback = """
        F (
            S (
                C ( is_dying self high) ::
                S (
                    C (in_reach foe) ::
                    A (move away_from closest foe)
                )::
                S (
                    C (in_reach friend) ::
                    A (move away_from closest friend)
                )::
                A (move away_from closest foe)::
                A (move away_from closest friend)
            )::
            A (attack closest any) ::
            A (stand)
        )
        """
# -


handcrafted_bts = {
    "Stand": {"bt": "A (stand)", "description": "The units do nothing."},
    "Follow_map": {
        "bt": "A (follow_map toward)",
        "description": "The units follow their direction map (stand if not direction map)",
    },
    "Attack": {
        "bt": set_default(bt_attack_and_wait, "A (follow_map toward)"),
        "description": "The units attack all the enemies in range.",
    },
    "Attack_in_close_range": {
        "bt": set_default(bt_attack_and_chase, "A (follow_map toward)"),
        "description": "The units wait for their opponent, attack them and then move toward them.",
    },
    "Attack_in_long_range": {
        "bt": set_default(bt_attack_and_stay_out_of_range, "A (follow_map toward)"),
        "description": "Attack while keeping out of reach of the enemy.",
    },
    "Flee": {
        "bt": set_default(bt_flee, "A (follow_map toward)"),
        "description": "Move away from closest foe in sight.",
    },
    "Follow_allies": {
        "bt": "F (A (move toward closest foe) :: A (stand))",
        "description": "Move toward the closest foe in sight. (For the NPC civilians)",
    },
    "Defend": {"bt": bt_attack_and_wait, "description": "The units attack all the enemies in range without moving."},
    "Defend_contact": {
        "bt": bt_attack_and_chase,
        "description": "The units attack all the enemies in range without moving.",
    },
    "LR_out_of_forest": {
        "bt": f"F (S (C (is_in_forest) :: A (follow_map toward)) :: {bt_attack_and_stay_out_of_range})"
    },
    "SR_out_of_forest": {"bt": f"F (S (C (is_in_forest) :: A (follow_map toward)) :: {bt_attack_and_chase})"},
    "Test": {"bt": f"F (S (C (is_in_forest) :: A (move north)) :: A (move west))"},
}

LLM_BTs = {
    "stand": "Stand",
    "ignore_enemies": "Follow_map",
    "flee": "Flee",
    "attack_in_close_range": "Attack_in_close_range",
    "attack_static": "Attack",
    "attack_in_long_range": "Attack_in_long_range",
    "to_contact": "Follow_allies",
    "defend": "Defend",
    "defend_contact": "Defend_contact",
    "sr_out_of_forest": "SR_out_of_forest",
    "lr_out_of_forest": "LR_out_of_forest",
    "test": "Test",
}
# -


def bt_fn(bt_str):
    dsl_tree = btc2sim.dsl.parse(btc2sim.dsl.read(bt_str))
    bt = btc2sim.bt.seed_fn(dsl_tree)
    return lambda obs, env_info, agent_info, rng: bt(obs, env_info, agent_info, rng)[1]


def bt_bank_jit():
    bts_bank = {}
    bts_bank_variant_subsets = {}
    for key, bt in handcrafted_bts.items():
        subsets, variants = compute_all_variants(bt["bt"], unit_types)
        bts_bank_variant_subsets[key] = subsets
        for i, bt_txt in enumerate(variants):
            name = key + (("_" + str(i)) if len(variants) > 1 else "")
            bts_bank[name] = bt_fn(bt_txt)
    return bts_bank, bts_bank_variant_subsets


def find_bt_key(bt_name, subset, bts_bank_variant_subsets):
    if len(bts_bank_variant_subsets[bt_name]) == 1:
        return bt_name
    else:
        return bt_name + "_" + str(bts_bank_variant_subsets[bt_name].index(subset))
