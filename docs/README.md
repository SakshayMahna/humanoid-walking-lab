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

```bash
.venv/bin/python docs/_scripts/week2_assets.py robot
```

Week 2 needs the policies in `3p/` (see the main [README](../README.md#third-party-repositories)).

## Posts

| Week | Post |
|---|---|
| 1 | [What a Pendulum Taught Me About Humanoid Control](https://sakshaymahna.github.io/humanoid-walking-lab/blog/2026/10/10/week-1.html) |
| 2 | [Making the G1 Walk](https://sakshaymahna.github.io/humanoid-walking-lab/blog/2026/10/15/week-2.html) |
