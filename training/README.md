# Training on Kaggle

Training our own G1 walking policies with [mjlab](https://github.com/mujocolab/mjlab) on Kaggle's free GPUs
(2 × Tesla T4), driven from this repository with the Kaggle CLI, then evaluated with the same MuJoCo and Gazebo
tools as the reference policies.

## Layout

| Path | What it is |
|---|---|
| [`kaggle/teaching-a-unitree-g1-to-walk-reinforcement-learn.ipynb`](kaggle/teaching-a-unitree-g1-to-walk-reinforcement-learn.ipynb) | The tutorial notebook: machine check, install, (speed test), training, output listing. All run settings in the Settings cell |
| [`kaggle/kernel-metadata.json`](kaggle/kernel-metadata.json) | Kaggle settings for pushes: private, GPU, internet, `machine_shape: NvidiaTeslaT4` (gives two T4s in batch runs) |
| `kaggle/output/v<N>/` | Downloaded outputs of notebook version N (not tracked: checkpoints are large) |
| [`evaluate_policy.py`](evaluate_policy.py) | MuJoCo test table for trained `policy.onnx` files next to Unitree's policy: stand, forward 0.2-1.0, sideways, turn |
| [`render_policy.py`](render_policy.py) | MuJoCo video: the same command sequence for every policy, side by side |
| [`gazebo_compare.sh`](gazebo_compare.sh) | Gazebo recordings (corrected controller + armature, sim clock) of trained runs next to Unitree's policy |

## Workflow

```bash
.venv/bin/kaggle auth login                       # once (browser login; nothing stored in the repo)
.venv/bin/kaggle kernels push -p training/kaggle  # runs the whole notebook on Kaggle (uses GPU quota)
.venv/bin/kaggle kernels status sakshaymahna/teaching-a-unitree-g1-to-walk-reinforcement-learn
.venv/bin/kaggle kernels output sakshaymahna/teaching-a-unitree-g1-to-walk-reinforcement-learn -p training/kaggle/output/v<N>

.venv/bin/python training/evaluate_policy.py training/kaggle/output/v<N>/logs/rsl_rl/g1_velocity/*/policy.onnx
.venv/bin/python training/render_policy.py training/kaggle/output/v<N>/logs/rsl_rl/g1_velocity/*/policy.onnx
training/gazebo_compare.sh training/kaggle/output/v<N>/logs/rsl_rl/g1_velocity/<run> [<run> ...]
```

A push runs the notebook top to bottom, unattended; the output (logs, checkpoints every 100 iterations, TensorBoard
files, `policy.onnx`) is only available when the run ends. Gazebo loads a trained policy through
`policy.launch.py policy:=unitree policy_file:=<path>` (same 98-number observation as Unitree's policy).

## Setup lessons

- Pins beyond Unitree's `setup.py`: `mujoco==3.5.0` (newer MuJoCo drops `mjENBL_MULTICCD`, which MuJoCo Warp 3.5.0
  uses) and `warp-lang==1.12.0` (newer Warp removed `warp.context`, which mjlab 1.2.0 uses).
- Multi-GPU flag format for this mjlab: `--gpu-ids "[0, 1]"` (Unitree's README shows `0 1`, which doesn't parse).
- A notebook saved without a GPU session is recorded as a CPU notebook (pinned CPU `docker_image`): the first push
  ran without CUDA. Fix: no `docker_image` in the metadata, GPU and internet on in the notebook's own Kaggle settings.
- Kaggle CLI 2.x logs in with `kaggle auth login` (or a token in `~/.kaggle/access_token`), not `kaggle.json`.

## Results

### Speed (T4, 4096 robots, `Unitree-G1-Flat`)

About 5.1 s per iteration (4.6 s simulation, 0.5 s learning), about 19,000 simulation steps per second per GPU,
plus about 2 minutes of start-up (MuJoCo Warp compiling). Two runs at once, one per GPU, don't slow each other
down. 3000 iterations: about 4.4 h.

### Baseline (notebook version 4): `Unitree-G1-Flat`, 3000 iterations, seeds 1 and 2

Learning: upright by about iteration 300 (episodes reach the 1000-step limit), following commands by about 1000,
mostly done by 2000 (reward 38 → 39-40 from 2000 to 3000). Unitree's command curriculum widens the speed range only
at iteration 5000, so these policies were trained on −0.5…1.0 m/s forward, ±0.5 sideways, ±1 rad/s turning.

MuJoCo ([`evaluate_policy.py`](evaluate_policy.py), 12 s per command):

| | Stand | vx 0.2 | vx 0.3 | vx 0.5 | vx 1.0 | vy 0.3 | wz 0.5 |
|---|---|---|---|---|---|---|---|
| Unitree's policy | 0.01 m | 0.22 | 0.31 | 0.50 | 1.02 | 0.24 | 0.53 |
| Ours, seed 1 | 0.01 m | 0.18 | 0.27 | 0.45 | 1.03 | 0.24 | 0.46 |
| Ours, seed 2 | 0.02 m | 0.19 | 0.28 | 0.46 | 0.99 | 0.24 | 0.44 |

Gazebo (corrected controller + armature, sim clock, one recorded run each): all three stand perfectly still for
30 s; walking at 0.5: Unitree 0.50 m/s, ours 0.45 / 0.46 m/s (their MuJoCo speed).

Heading drift while walking at 0.5 (°/s) is a property of each trained policy and differs between seeds:

| | MuJoCo 5 ms | MuJoCo 1 ms | Gazebo |
|---|---|---|---|
| Unitree's policy | +0.72 | +1.03 | +0.71 |
| Ours, seed 1 | +0.47 | +0.58 | +1.21 |
| Ours, seed 2 | −1.91 | −1.89 | −3.14 |

So a single seed cannot show a heading effect when comparing variants.

Videos (not tracked): `recordings/training/mujoco_compare.mp4`, `gazebo_compare_{stand,walk}.mp4`.

## Next

Writing the task ourselves in the notebook (robot and actions, observations, commands, rewards one term at a
time, terminations, randomization), trained with mjlab's PPO, and compared term by term with `Unitree-G1-Flat`;
plus a chapter reading `rsl_rl`'s PPO.
