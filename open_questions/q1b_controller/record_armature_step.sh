#!/usr/bin/env bash
# Videos for Q1b experiments 3-4 (run on the Mac): Diego's original controller unchanged, Unitree's policy,
# sim clock; baseline vs + armature vs 0.2 ms step, standing and walking. Then labelled side-by-sides.
# Output: recordings/q1b/rec_<setup>_<scenario>.{mp4,csv}, recordings/q1b/armature_step_{stand,walk}.mp4
# The videos run on wall time: the 0.2 ms runs are ~0.6x real time, so they look slower.
set -euo pipefail
cd "$(dirname "$0")/../.."
.venv/bin/python gazebo/make_gain_variants.py > /dev/null
[ -f recordings/gz_models/worlds/empty_0p2ms.sdf ] || { echo "run gazebo_armature_step.sh first (writes the 0.2 ms world)"; exit 1; }
L=/home/ubuntu/humanoid-walking-lab/recordings/gz_models
args_for() {
    case "$1" in
        baseline)   echo "robot_model:=g1_free_roam" ;;
        armature)   echo "robot_model:=$L/original_armature/model.sdf" ;;
        step_0p2ms) echo "robot_model:=g1_free_roam world:=$L/worlds/empty_0p2ms.sdf" ;;
    esac
}
for setup in baseline armature step_0p2ms; do
    gazebo/record.sh 0 0 0 30 30 "q1b/rec_${setup}_stand" "policy:=unitree clock:=sim $(args_for $setup)" | grep -E "^Done|rror" || true
    gazebo/record.sh 0.5 0 0 24 4 "q1b/rec_${setup}_walk" "policy:=unitree clock:=sim $(args_for $setup)" | grep -E "^Done|rror" || true
done

Q=recordings/q1b
for scenario in stand walk; do
    seconds=30; what="zero command for 30 s"
    [ "$scenario" = walk ] && seconds=24 && what="stand 4 s, then walk at 0.5 m/s"
    .venv/bin/python gazebo/side_by_side.py "$Q/armature_step_$scenario.mp4" --seconds 120 \
        --title "Gazebo, original controller unchanged ($what)" \
        "Original (no armature, 1 ms step)|$Q/rec_baseline_$scenario.mp4|$Q/rec_baseline_$scenario.csv" \
        "A: + armature in the links|$Q/rec_armature_$scenario.mp4|$Q/rec_armature_$scenario.csv" \
        "B: 0.2 ms physics step (slower)|$Q/rec_step_0p2ms_$scenario.mp4|$Q/rec_step_0p2ms_$scenario.csv"
done
