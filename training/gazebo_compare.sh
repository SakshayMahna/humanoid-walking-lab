#!/usr/bin/env bash
# Our trained policies in Gazebo next to Unitree's (run on the Mac). Best settings from the open questions:
# corrected controller + armature in the link inertias (mj_gains_vel_d_armature), sim clock. Recorded, standing
# (zero command, 30 s) and walking (stand 4 s, then vx = 0.5), then labelled side-by-side videos.
# Usage: training/gazebo_compare.sh RUN_DIR [RUN_DIR ...]   (folders with a policy.onnx, under the repo)
# Output: recordings/training/gz_<label>_<scenario>.{mp4,csv}, recordings/training/gazebo_compare_{stand,walk}.mp4
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python gazebo/make_gain_variants.py > /dev/null
REPO_IN_CONTAINER=/home/ubuntu/humanoid-walking-lab
MODEL=$REPO_IN_CONTAINER/recordings/gz_models/mj_gains_vel_d_armature/model.sdf
OUT=recordings/training

labels="unitree"
for run in "$@"; do labels="$labels $(basename "$run")"; done

record() {   # LABEL POLICY_FILE_ARG
    local label=$1 extra=$2
    gazebo/record.sh 0 0 0 30 30 "training/gz_${label}_stand" "policy:=unitree clock:=sim robot_model:=$MODEL $extra" | grep -E "^Done|rror" || true
    gazebo/record.sh 0.5 0 0 24 4 "training/gz_${label}_walk" "policy:=unitree clock:=sim robot_model:=$MODEL $extra" | grep -E "^Done|rror" || true
}
record unitree ""
for run in "$@"; do
    [ -f "$run/policy.onnx" ] || { echo "no policy.onnx in $run"; exit 1; }
    record "$(basename "$run")" "policy_file:=$REPO_IN_CONTAINER/$run/policy.onnx"
done

for scenario in stand walk; do
    seconds=30; what="zero command for 30 s"
    [ "$scenario" = walk ] && seconds=24 && what="stand 4 s, then walk at 0.5 m/s"
    tiles=()
    for label in $labels; do
        name="$label"; [ "$label" = unitree ] && name="Unitree's policy"
        case "$label" in *seed*) name="Ours: ${label##*_}";; esac
        tiles+=("$name|$OUT/gz_${label}_$scenario.mp4|$OUT/gz_${label}_$scenario.csv")
    done
    .venv/bin/python gazebo/side_by_side.py "$OUT/gazebo_compare_$scenario.mp4" --seconds 120 \
        --title "Gazebo, corrected controller + armature, sim clock ($what)" "${tiles[@]}"
done
