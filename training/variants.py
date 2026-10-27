"""Training variants of Unitree's G1 walking task: each one changes one thing in the baseline config.

A variant is a function that takes the baseline environment config (Unitree-G1-Flat, a mutable mjlab dataclass) and
changes it in place. `train_variant.py` applies one and trains it.

  baseline           Unitree-G1-Flat unchanged
  ladder_<k>         "add one in": only the reward terms of ladder steps 0..k (PLAN.md, Track 1, Days 2-3)
  loo_<term>         "leave one out": the baseline without one reward term
  ladder_2_plus_<t>  ladder step 2 plus one of step 3's terms (action_rate_l2 or joint_acc_l2)
  obs_no_clock       the full reward set minus foot_gait, and the policy without its gait clock input
  obs_lin_vel        the full set, the policy also observing the base linear velocity (walk.onnx's design)
  dr_armature        the full set, joint armature randomized per robot between 0 and its full value
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


def ladder_2_plus(term):
    """Ladder step 2 plus one smoothness term: which of step 3's two terms makes the robot stand up?"""
    def variant(cfg):
        baseline(cfg)
        keep_rewards(cfg, [t for step in LADDER[:3] for t in step] + [term])
    return variant


def obs_no_clock(cfg):
    """The policy without its gait clock input ("phase"). foot_gait is defined by the same clock, so it goes too.
    The critic keeps the clock (it only guides training; the deployed policy never sees it)."""
    baseline(cfg)
    del cfg.observations["actor"].terms["phase"]
    del cfg.rewards["foot_gait"]


def obs_lin_vel(cfg):
    """The policy also gets the base linear velocity (as walk.onnx does), with the critic's noise (+-0.5 m/s)."""
    import copy
    baseline(cfg)
    actor = cfg.observations["actor"].terms
    term = copy.deepcopy(cfg.observations["critic"].terms["base_lin_vel"])
    cfg.observations["actor"].terms = {"base_lin_vel": term, **actor}     # first, like walk.onnx


def dr_armature(cfg):
    """Each robot's joint armature scaled by a random factor in [0, 1] at startup: from none (as in Gazebo) to the
    full motor value. Tests whether training on varied joint dynamics fixes the Gazebo-without-armature failure."""
    from mjlab.envs.mdp import dr
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    baseline(cfg)
    cfg.events["armature"] = EventTermCfg(mode="startup", func=dr.joint_armature,
                                          params={"asset_cfg": SceneEntityCfg("robot"), "operation": "scale",
                                                  "ranges": (0.0, 1.0)})


VARIANTS = {"baseline": baseline}
VARIANTS.update({f"ladder_{k}": ladder(k) for k in range(len(LADDER))})
VARIANTS.update({f"loo_{term}": leave_out(term) for term in ALL_TERMS})
VARIANTS.update({f"ladder_2_plus_{term}": ladder_2_plus(term) for term in LADDER[3]})
VARIANTS.update({"obs_no_clock": obs_no_clock, "obs_lin_vel": obs_lin_vel, "dr_armature": dr_armature})
