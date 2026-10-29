# Blog

Source of the weekly blog at **[sakshaymahna.github.io/humanoid-walking-lab](https://sakshaymahna.github.io/humanoid-walking-lab/)**.
Jekyll with the [Minimal Mistakes](https://github.com/mmistakes/minimal-mistakes) theme (dark skin), published by
GitHub Pages from this `docs/` folder on `main`.

| Path | Contents |
|---|---|
| [`_posts/`](_posts/) | One Markdown file per post |
| [`_config.yml`](_config.yml) | Site settings and theme |
| [`_includes/head/custom.html`](_includes/head/custom.html) | MathJax and the heading font |
| [`assets/css/main.scss`](assets/css/main.scss) | Theme overrides (centred images and tables, heading font) |
| [`assets/images/week-N/`](assets/images/) | Images, GIFs and the banner for each post |
| [`assets/videos/week-N/`](assets/videos/) | LinkedIn videos and GIFs for each week |
| [`_scripts/`](_scripts/) | Scripts that generate those assets |

## Generating the assets

The scripts reuse the code in [`exercises/`](../exercises/), so the animations show the same physics as the
exercises; only colours, lighting and layout are added. Run them from the repo root:

| Script | Output |
|---|---|
| `week1_assets.py` | `assets/images/week-1/`: pendulum and PD-control figures and GIFs |
| `week1_banner.py [preview_dir]` | `assets/images/week-1/banner.png` (optional preview with a mock title) |
| `week1_video.py` | `assets/videos/week-1/pd_sweeps.mp4`, LinkedIn video (needs `ffmpeg`) |
| `week2_assets.py [planetary] [phase] [robot] [banner]` | `assets/images/week-2/`: gearbox, gait clock and robot GIFs, banner |
| `week2_linkedin.py` | `assets/videos/week-2/policy_compare.gif` and `banner.png`: both policies on four commands |
| `week3_assets.py [banner] [diagram] [kick] [damping] [grid] [gazebo] [variants]` | `assets/images/week-3/` (banner, SVG charts and diagram, GIFs) and `assets/videos/week-3/` (Gazebo clips as MP4) |
| `week3_runs.sh` | The sim-clock Gazebo runs behind the Week 3 tables → `recordings/week3/` (needs Docker) |
| `week4_banner.py [preview_dir]` | `assets/images/week-4/banner.png`: the reward ladder (lying → walking) as ghosts of the real trained policies |
| `week4_videos.py [clip ...] [--rerender]` | `assets/videos/week-4/`: training-run clips in plain MuJoCo, same camera and test for every robot (needs the Kaggle run outputs); the Gazebo tables and clips come from `training/gazebo_repeats.sh` |

```bash
.venv/bin/python docs/_scripts/week2_assets.py robot
```

Weeks 2 and 3 need the policies in `3p/` (see the main [README](../README.md#third-party-repositories)); Week 3's Gazebo videos also need the recordings made by the `gazebo/` and `open_questions/` scripts.

Post URLs come from the file name (`permalink: /blog/:title/`), so name new posts after their title, e.g. `2026-10-31-training-a-walking-policy.md`. Old week-numbered URLs redirect to the new ones (`redirect_from` in each post).

## Posts

| Week | Post |
|---|---|
| 1 | [What a Pendulum Taught Me About Humanoid Control](https://sakshaymahna.github.io/humanoid-walking-lab/blog/what-a-pendulum-taught-me-about-humanoid-control/) |
| 2 | [Making the G1 Walk](https://sakshaymahna.github.io/humanoid-walking-lab/blog/making-the-unitree-g1-walk/) |
| 3 | [From MuJoCo to Gazebo](https://sakshaymahna.github.io/humanoid-walking-lab/blog/from-mujoco-to-gazebo/) |
| 4 | [What Each Reward Teaches a Walking Robot](https://sakshaymahna.github.io/humanoid-walking-lab/blog/what-each-reward-teaches-a-walking-robot/) |
