#!/usr/bin/env bash
# Q1 test 1 videos (run on the Mac): MuJoCo | Gazebo, original controller | Gazebo, corrected controller, per repeat,
# plus an overview with all repeats stacked. Needs the runs from standing_test.sh and standing_reference.py.
# Output: recordings/standing/compare_<n>.mp4, recordings/standing/compare_all.mp4
set -euo pipefail
REPEATS=${1:-3}
cd "$(dirname "$0")/../.."
S=recordings/standing
PY=.venv/bin/python

rows=()
for n in $(seq "$REPEATS"); do
    $PY gazebo/side_by_side.py "$S/compare_$n.mp4" \
        --title "Unitree policy, zero command for 30 s (Gazebo repeat $n)" \
        "MuJoCo (training simulator)|$S/stand_mujoco_unitree_1.mp4|$S/stand_mujoco_unitree_1.csv" \
        "Gazebo - original controller|$S/stand_original_unitree_$n.mp4|$S/stand_original_unitree_$n.csv" \
        "Gazebo - corrected controller|$S/stand_corrected_unitree_$n.mp4|$S/stand_corrected_unitree_$n.csv"
    rows+=(-i "$S/compare_$n.mp4")
done

ffmpeg -loglevel error -y "${rows[@]}" -filter_complex "vstack=inputs=$REPEATS" \
    -c:v libx264 -pix_fmt yuv420p "$S/compare_all.mp4"
echo "wrote $S/compare_all.mp4"
