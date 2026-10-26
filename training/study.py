"""Build the generated part of the reward study: learning curves per reward term and the evaluation table.

Reads every downloaded run (training/kaggle/output/v*/logs/rsl_rl/g1_velocity/<date>_<variant>_seed<n>/ with its
training log training/kaggle/output/v*/train_<variant>_seed<n>.log) and writes, in training/reward_study/:

  charts/overview.svg     mean reward and episode length over training, one line per run
  charts/terms.svg        every reward term's value over training, one panel per term, one line per run that has it
  results.md              runs (variant, terms, seed, iterations, notebook version) and report.py's evaluation table

    .venv/bin/python training/study.py [--no-eval]     (--no-eval: skip the MuJoCo evaluation, charts only)
"""

import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))
from variants import ALL_TERMS, LADDER  # noqa: E402

OUTPUT = REPO / "training" / "kaggle" / "output"
STUDY = REPO / "training" / "reward_study"

# Validated categorical palette (dark surface #121421): ladder steps 0-7 in fixed order; the full reward set
# (baseline = ladder step 8) is a neutral reference line.
PALETTE = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
BASELINE_COLOR = "#ebebf5"
BG, TEXT, MUTED, GRID = "#121421", "#ebebf5", "#a3a6c0", "#2c3048"


# ---------------------------------------------------------------------------
# Runs and their training logs
# ---------------------------------------------------------------------------

def find_runs():
    """All downloaded runs: dict label -> info (variant, seed, version, dir, log, iterations)."""
    runs = {}
    for run_dir in sorted(OUTPUT.glob("v*/logs/rsl_rl/g1_velocity/*_seed*")):
        name = run_dir.name.split("_", 2)[2]                      # <variant>_seed<n>
        variant, seed = name.rsplit("_seed", 1)
        version = run_dir.parents[3].name
        log = run_dir.parents[3] / f"train_{name}.log"
        if not log.exists():
            continue
        curves = parse_log(log)
        if curves["iterations"] < 100:                            # pipeline tests, not experiments
            continue
        label = f"{name} ({curves['iterations']} it)"
        runs[label] = {"variant": variant, "seed": int(seed), "version": version, "dir": run_dir, "log": log,
                       "curves": curves, "iterations": curves["iterations"]}
    return runs


def parse_log(path):
    """Per-iteration values from an rsl_rl training log: mean reward, episode length and every Episode_Reward term."""
    text = path.read_text()
    blocks = re.split(r"Learning iteration (\d+)/\d+", text)[1:]
    data = {}
    for it, block in zip(blocks[0::2], blocks[1::2]):
        values = dict(re.findall(r"^\s*([\w/]+(?: \w+)*):\s+(-?[\d.]+(?:e-?\d+)?)\s*$", block, flags=re.M))
        for key, value in values.items():
            if key in ("Mean reward", "Mean episode length") or key.startswith("Episode_Reward/"):
                data.setdefault(key.replace("Episode_Reward/", ""), []).append((int(it), float(value)))
    curves = {k: np.array(v) for k, v in data.items()}
    curves["iterations"] = int(max(int(i) for i in blocks[0::2]) + 1) if blocks else 0
    return curves


def variant_order(variant):
    if variant.startswith("ladder_"):
        return (0, int(variant.split("_")[1]))
    if variant == "baseline":
        return (0, len(LADDER) - 1)
    return (1, variant)


def color_of(variant):
    if variant == "baseline":
        return BASELINE_COLOR
    if variant.startswith("ladder_"):
        return PALETTE[int(variant.split("_")[1]) % len(PALETTE)]
    return PALETTE[hash(variant) % len(PALETTE)]


# ---------------------------------------------------------------------------
# SVG charts
# ---------------------------------------------------------------------------

def text(x, y, s, size=12, color=TEXT, anchor="start", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def panel(x0, y0, w, h, title, series, xmax, y_label=""):
    """One small line chart. series: list of (label, color, xs, ys, dashed)."""
    out = [text(x0, y0 - 8, title, 13, TEXT, weight="bold")]
    ys_all = np.concatenate([s[3] for s in series]) if series else np.array([0.0, 1.0])
    lo, hi = float(min(ys_all.min(), 0.0)), float(max(ys_all.max(), 0.0))
    if hi - lo < 1e-9:
        hi = lo + 1.0
    ticks = nice_ticks(lo, hi)
    lo, hi = min(lo, ticks[0]), max(hi, ticks[-1])
    px0, px1, py0, py1 = x0 + 46, x0 + w - 8, y0 + 6, y0 + h - 22
    for t in ticks:
        y = py1 - (t - lo) / (hi - lo) * (py1 - py0)
        zero = abs(t) < 1e-12
        out.append(f'<line x1="{px0}" y1="{y:.1f}" x2="{px1}" y2="{y:.1f}" stroke="{"#4a4f6e" if zero else GRID}" '
                   f'stroke-width="1"/>')
        out.append(text(px0 - 5, y + 4, f"{t:g}", 10, MUTED, "end"))
    for t in np.linspace(0, xmax, 5):
        x = px0 + t / xmax * (px1 - px0)
        out.append(text(x, py1 + 15, f"{t:.0f}", 10, MUTED, "middle"))
    for label, color, xs, ys, dashed in series:
        keep = xs <= xmax
        step = max(1, int(keep.sum() // 300))                       # at most ~300 points per line
        pts = " ".join(f"{px0 + x / xmax * (px1 - px0):.1f},{py1 - (y - lo) / (hi - lo) * (py1 - py0):.1f}"
                       for x, y in zip(xs[keep][::step], smooth(ys[keep])[::step]))
        dash = ' stroke-dasharray="5,4"' if dashed else ""
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
    if y_label:
        out.append(text(px0, py0 + 12, y_label, 10, MUTED))
    return out


def nice_ticks(lo, hi, n=4):
    """Round tick values (1, 2, 2.5 or 5 times a power of ten) covering lo..hi."""
    span = hi - lo
    raw = span / n
    mag = 10 ** np.floor(np.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = np.floor(lo / step) * step
    ticks = np.arange(start, hi + step * 0.999, step)
    return [float(np.round(t, 10)) for t in ticks]


def smooth(y, k=15):
    """Moving average over k iterations (the logged values are noisy, per-iteration episode means)."""
    if len(y) < k:
        return y
    kernel = np.ones(k) / k
    padded = np.concatenate([np.full(k // 2, y[0]), y, np.full(k - 1 - k // 2, y[-1])])
    return np.convolve(padded, kernel, mode="valid")


def legend(x, y, items, width):
    out, cx, cy = [], x, y
    for label, color, dashed in items:
        w = 34 + 6.6 * len(label)
        if cx + w > x + width:
            cx, cy = x, cy + 20
        dash = ' stroke-dasharray="5,4"' if dashed else ""
        out.append(f'<line x1="{cx}" y1="{cy - 4}" x2="{cx + 22}" y2="{cy - 4}" stroke="{color}" stroke-width="3"{dash}/>')
        out.append(text(cx + 28, cy, label, 12))
        cx += w + 14
    return out, cy


def svg(width, height, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'font-family="Helvetica, Arial, sans-serif">\n<rect width="100%" height="100%" fill="{BG}" rx="8"/>\n'
            + "\n".join(body) + "\n</svg>\n")


def series_for(runs, key, xmax):
    out = []
    for label, r in sorted(runs.items(), key=lambda kv: (variant_order(kv[1]["variant"]), kv[1]["iterations"])):
        c = r["curves"].get(key)
        if c is None:
            continue
        out.append((label, color_of(r["variant"]), c[:, 0], c[:, 1], r["iterations"] > xmax))
    return out


def charts(runs, xmax):
    (STUDY / "charts").mkdir(parents=True, exist_ok=True)
    items = [(label, color_of(r["variant"]), r["iterations"] > xmax)
             for label, r in sorted(runs.items(), key=lambda kv: (variant_order(kv[1]["variant"]), kv[1]["iterations"]))]
    W = 1000
    leg, ly = legend(30, 64, items, W - 60)
    top = ly + 40
    body = [text(30, 34, "Learning curves: mean reward and episode length", 18, TEXT, weight="bold"),
            text(30, 50, f"first {xmax} iterations; dashed = run trained longer (shown up to {xmax}); "
                         "15-iteration moving average", 12, MUTED)] + leg
    body += panel(30, top, 460, 260, "Mean reward per episode", series_for(runs, "Mean reward", xmax), xmax)
    body += panel(510, top, 460, 260, "Mean episode length (steps; 1000 = 20 s, no fall)",
                  series_for(runs, "Mean episode length", xmax), xmax)
    body.append(text(30, top + 290, "x axis: training iteration", 11, MUTED))
    (STUDY / "charts" / "overview.svg").write_text(svg(W, top + 305, body))

    cols, pw, ph = 4, 240, 190
    leg, ly = legend(30, 64, items, cols * pw - 20)
    top = ly + 40
    body = [text(30, 34, "Every reward term over training (per-episode contribution, weighted)", 18, TEXT, weight="bold"),
            text(30, 50, f"one panel per term, one line per run that has the term; first {xmax} iterations; "
                         "dashed = run trained longer", 12, MUTED)] + leg
    for k, term in enumerate(ALL_TERMS):
        x0, y0 = 20 + (k % cols) * pw, top + (k // cols) * (ph + 26)
        body += panel(x0, y0, pw - 10, ph, term, series_for(runs, term, xmax), xmax)
    height = top + ((len(ALL_TERMS) + cols - 1) // cols) * (ph + 26) + 10
    (STUDY / "charts" / "terms.svg").write_text(svg(cols * pw + 20, height, body))


# ---------------------------------------------------------------------------
# results.md
# ---------------------------------------------------------------------------

def results(runs, evaluate):
    lines = ["# Reward study: generated results", "",
             "Generated by [`training/study.py`](../study.py) from the downloaded runs; do not edit by hand. "
             "The analysis is in [README.md](README.md).", "",
             "## Runs", "", "| Run | Reward terms | Notebook version | Iterations |", "|---|---|---|---|"]
    ordered = sorted(runs.items(), key=lambda kv: (variant_order(kv[1]["variant"]), kv[1]["iterations"]))
    for label, r in ordered:
        terms = [t for t in ALL_TERMS if t in r["curves"]]
        lines.append(f"| {label} | {len(terms)}: {', '.join(terms)} | {r['version']} | {r['iterations']} |")
    lines += ["", "## Learning curves", "", "![Mean reward and episode length](charts/overview.svg)", "",
              "![Every reward term over training](charts/terms.svg)", ""]
    if evaluate:
        import report
        table = report.run_report([r["dir"] for _, r in ordered], labels=[label for label, _ in ordered])
        lines += ["## Evaluation in plain MuJoCo ([`report.py`](../report.py))", "",
                  "Unitree's model, 5 ms physics, policy every 20 ms. \"fell\" = pelvis below 0.4 m; \"unstable\" = "
                  "the policy drove MuJoCo's simulation to infinite accelerations.", "", table, ""]
    (STUDY / "results.md").write_text("\n".join(lines))


def main():
    runs = find_runs()
    xmax = 1000
    charts(runs, xmax)
    results(runs, evaluate="--no-eval" not in sys.argv)
    print(f"{len(runs)} runs -> {STUDY.relative_to(REPO)}/results.md, charts/overview.svg, charts/terms.svg")


if __name__ == "__main__":
    main()
