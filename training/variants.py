"""Training variants of Unitree's G1 walking task: each one changes one thing in the baseline config.

A variant is a function that takes the baseline environment config (Unitree-G1-Flat, a mutable mjlab dataclass) and
changes it in place. `train_variant.py` applies one and trains it.

  baseline           Unitree-G1-Flat unchanged
  ladder_<k>         "add one in": only the reward terms of ladder steps 0..k (PLAN.md, Track 1, Days 2-3)
  loo_<term>         "leave one out": the baseline without one reward term
"""

# Reward terms of Unitree-G1-Flat, grouped into the add-one-in ladder (each step keeps the previous ones)
LADDER = [
    ["track_linear_velocity", "track_angular_velocity", "is_terminated"],   # 0: follow the command
    ["body_orientation_l2"],                                               # 1: upright
    ["pose"],                                                              # 2: near the default posture
    ["action_rate_l2", "joint_acc_l2"],                                    # 3: smoothness
    ["foot_gait"],                                                         # 4: alternating steps
    ["foot_clearance", "foot_slip", "soft_landing"],                       # 5: feet
    ["stand_still"],                                                       # 6: stand still at zero command
    ["body_ang_vel", "angular_momentum"],                                  # 7: torso wobble
    ["joint_pos_limits", "self_collisions"],                               # 8: joint range, self-collisions
]
ALL_TERMS = [term for step in LADDER for term in step]


def keep_rewards(cfg, terms):
    """Keep only these reward terms (they must all exist in the config)."""
    missing = set(terms) - set(cfg.rewards)
    assert not missing, f"unknown reward terms: {missing}"
    for name in list(cfg.rewards):
        if name not in terms:
            del cfg.rewards[name]


def baseline(cfg):
    assert set(cfg.rewards) == set(ALL_TERMS), f"baseline rewards changed: {sorted(cfg.rewards)}"


def ladder(k):
    def variant(cfg):
        baseline(cfg)
        keep_rewards(cfg, [term for step in LADDER[:k + 1] for term in step])
    return variant


def leave_out(term):
    def variant(cfg):
        baseline(cfg)
        keep_rewards(cfg, [t for t in ALL_TERMS if t != term])
    return variant


VARIANTS = {"baseline": baseline}
VARIANTS.update({f"ladder_{k}": ladder(k) for k in range(len(LADDER))})
VARIANTS.update({f"loo_{term}": leave_out(term) for term in ALL_TERMS})
