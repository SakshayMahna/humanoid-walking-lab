#!/usr/bin/env bash
# Q2 video (run on the Mac): the corrected controller with and without armature, Unitree's policy vs walk.onnx,
# in Gazebo, sim clock. Records Unitree's two setups (walk.onnx's come from gazebo_record.sh), then 2x2 grids:
# rows = policy, columns = without / with armature.
# Output: recordings/q2/gz_unitree_<setup>_<scenario>.{mp4,csv}, recordings/q2/policy_compare_{stand,walk}.mp4
set -euo pipefail
cd "$(dirname "$0")/../.."
Q=recordings/q2
for f in gz_corrected_stand gz_corrected_walk gz_corrected_armature_stand gz_corrected_armature_walk; do
    [ -f "$Q/$f.mp4" ] || { echo "missing $Q/$f.mp4: run gazebo_record.sh first"; exit 1; }
done
.venv/bin/python gazebo/make_gain_variants.py > /dev/null
L=/home/ubuntu/humanoid-walking-lab/recordings/gz_models
for setup in corrected corrected_armature; do
    model=$L/mj_gains_vel_d/model.sdf
    [ "$setup" = corrected_armature ] && model=$L/mj_gains_vel_d_armature/model.sdf
    gazebo/record.sh 0 0 0 30 30 "q2/gz_unitree_${setup}_stand" "policy:=unitree clock:=sim robot_model:=$model" | grep -E "^Done|rror" || true
    gazebo/record.sh 0.5 0 0 24 4 "q2/gz_unitree_${setup}_walk" "policy:=unitree clock:=sim robot_model:=$model" | grep -E "^Done|rror" || true
done

for scenario in stand walk; do
    seconds=30; what="zero command for 30 s"
    [ "$scenario" = walk ] && seconds=24 && what="stand 4 s, then walk at 0.5 m/s"
    .venv/bin/python gazebo/side_by_side.py "$Q/policy_compare_$scenario.mp4" --seconds 120 --cols 2 \
        --title "Gazebo, corrected controller, sim clock ($what)" \
        "Unitree: corrected, no armature|$Q/gz_unitree_corrected_$scenario.mp4|$Q/gz_unitree_corrected_$scenario.csv" \
        "Unitree: corrected + armature|$Q/gz_unitree_corrected_armature_$scenario.mp4|$Q/gz_unitree_corrected_armature_$scenario.csv" \
        "walk.onnx: corrected, no armature|$Q/gz_corrected_$scenario.mp4|$Q/gz_corrected_$scenario.csv" \
        "walk.onnx: corrected + armature|$Q/gz_corrected_armature_$scenario.mp4|$Q/gz_corrected_armature_$scenario.csv"
done
