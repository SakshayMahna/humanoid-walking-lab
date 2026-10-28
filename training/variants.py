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
  dr_armature_height dr_armature, and a robot whose pelvis drops below 0.5 m has fallen (no sitting)
  ladder_0_strict, baseline_strict   step 0 / full set, episode ends on pelvis or torso contact, -10 height penalty
  track_hard         full set + Humanoid-Gym's track_vel_hard and low_speed
  loo_foot_gait_track_hard  the same without foot_gait (does hard tracking remove the dead zone?)
  act_explicit       full set, explicit PD (mjlab IdealPdActuator) instead of MuJoCo's position actuator
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


def dr_armature_height(cfg):
    """dr_armature plus a height fall check. Unitree's task only ends an episode on a tilt over 70 degrees, so a robot
    sitting upright counts as standing; with randomized armature 2 of 3 runs settled into sitting (round 12)."""
    from mjlab.envs.mdp.terminations import root_height_below_minimum
    from mjlab.managers.termination_manager import TerminationTermCfg
    dr_armature(cfg)
    cfg.terminations["too_low"] = TerminationTermCfg(func=root_height_below_minimum,
                                                     params={"minimum_height": 0.5})   # stands at 0.78 m


STAND_HEIGHT = 0.78    # G1 pelvis height standing (m), unitree_rl_gym's base_height_target


def base_height_l2(env):
    """(pelvis height - standing height)^2, on flat ground (unitree_rl_gym's base_height, weight -10 there)."""
    return (env.scene["robot"].data.root_link_pos_w[:, 2] - STAND_HEIGHT) ** 2


def strict_fall(cfg):
    """Close the sitting loophole the way unitree_rl_gym does: the episode ends when the pelvis or torso touches
    the ground, and pelvis height away from standing is penalized (-10 (h - 0.78)^2: sitting costs ~4 per step)."""
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    from mjlab.tasks.velocity import mdp
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (ContactSensorCfg(
        name="body_ground_contact",
        primary=ContactMatch(mode="body", pattern=r"^(pelvis|torso_link)$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"), reduce="netforce", num_slots=1),)
    cfg.terminations["body_contact"] = TerminationTermCfg(func=mdp.illegal_contact,
                                                          params={"sensor_name": "body_ground_contact"})
    cfg.rewards["base_height"] = RewardTermCfg(func=base_height_l2, weight=-10.0)


def ladder_0_strict(cfg):
    """Tracking and the fall penalty only (ladder step 0, which lay down), with the strict fall check."""
    ladder(0)(cfg)
    strict_fall(cfg)


def baseline_strict(cfg):
    baseline(cfg)
    strict_fall(cfg)


def track_vel_hard(env):
    """Humanoid-Gym's track_vel_hard: a sharp exponential peak plus a linear term that never flattens out."""
    import torch
    robot, cmd = env.scene["robot"], env.command_manager.get_command("twist")
    lin = torch.norm(cmd[:, :2] - robot.data.root_link_lin_vel_b[:, :2], dim=1)
    ang = torch.abs(cmd[:, 2] - robot.data.root_link_ang_vel_b[:, 2])
    return (torch.exp(-10 * lin) + torch.exp(-10 * ang)) / 2 - 0.2 * (lin + ang)


def low_speed(env):
    """Humanoid-Gym's low_speed: -1 below half the forward command, +1.2 within 50-120 %, -2 the wrong way."""
    import torch
    vx = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    cx = env.command_manager.get_command("twist")[:, 0]
    reward = torch.zeros_like(vx)
    reward[vx.abs() < 0.5 * cx.abs()] = -1.0
    reward[(vx.abs() >= 0.5 * cx.abs()) & (vx.abs() <= 1.2 * cx.abs())] = 1.2
    reward[torch.sign(vx) != torch.sign(cx)] = -2.0
    return reward * (cx.abs() > 0.1)


def track_hard(cfg):
    """The full set plus Humanoid-Gym's two tracking terms (their weights): standing still when told to walk loses
    reward, and the tracking error keeps a gradient far from the target."""
    from mjlab.managers.reward_manager import RewardTermCfg
    baseline(cfg)
    cfg.rewards["track_vel_hard"] = RewardTermCfg(func=track_vel_hard, weight=0.5)
    cfg.rewards["low_speed"] = RewardTermCfg(func=low_speed, weight=0.2)


def loo_foot_gait_track_hard(cfg):
    """Without foot_gait (shuffle, dead zone) but with the hard tracking terms: do they remove the dead zone?"""
    track_hard(cfg)
    del cfg.rewards["foot_gait"]


def act_explicit(cfg):
    """Explicit PD instead of MuJoCo's position actuator: tau = kp (target - q) - kd q_dot computed every physics
    step and applied as a motor torque (as Gazebo's controller does). Same gains, limits and armature."""
    from mjlab.actuator import IdealPdActuatorCfg
    baseline(cfg)
    articulation = cfg.scene.entities["robot"].articulation
    articulation.actuators = tuple(
        IdealPdActuatorCfg(target_names_expr=a.target_names_expr, stiffness=a.stiffness, damping=a.damping,
                           effort_limit=a.effort_limit, armature=a.armature, frictionloss=a.frictionloss)
        for a in articulation.actuators)


VARIANTS = {"baseline": baseline}
VARIANTS.update({f"ladder_{k}": ladder(k) for k in range(len(LADDER))})
VARIANTS.update({f"loo_{term}": leave_out(term) for term in ALL_TERMS})
VARIANTS.update({f"ladder_2_plus_{term}": ladder_2_plus(term) for term in LADDER[3]})
VARIANTS.update({"obs_no_clock": obs_no_clock, "obs_lin_vel": obs_lin_vel, "dr_armature": dr_armature,
                 "dr_armature_height": dr_armature_height})
VARIANTS.update({"ladder_0_strict": ladder_0_strict, "baseline_strict": baseline_strict, "track_hard": track_hard,
                 "loo_foot_gait_track_hard": loo_foot_gait_track_hard, "act_explicit": act_explicit})
