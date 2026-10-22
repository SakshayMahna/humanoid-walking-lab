#!/usr/bin/env bash
# Q2 Test 2 (run on the Mac): walk.onnx in Gazebo with the same four setups as the MuJoCo grid, recorded,
# sim clock. Standing (zero command, 30 s) and walking (stand 4 s, then vx = 0.5), then labelled 2x2 grids.
#   original           Diego's model unchanged (Luis's controller, D on the error, no armature)
#   original_armature  + armature approximated in the link inertias
#   corrected          mj_gains_vel_d: policy gains and limits, D as joint damping (implicit), no armature
#   corrected_armature both fixes
# Output: recordings/q2/gz_<setup>_<scenario>.{mp4,csv}, recordings/q2/gazebo_grid_{stand,walk}.mp4
set -euo pipefail
cd "$(dirname "$0")/../.."
.venv/bin/python gazebo/make_gain_variants.py > /dev/null
L=/home/ubuntu/humanoid-walking-lab/recordings/gz_models
args_for() {   # a function: macOS bash 3.2 has no associative arrays
    case "$1" in
        original)           echo "robot_model:=g1_free_roam" ;;
        original_armature)  echo "robot_model:=$L/original_armature/model.sdf" ;;
        corrected)          echo "robot_model:=$L/mj_gains_vel_d/model.sdf" ;;
        corrected_armature) echo "robot_model:=$L/mj_gains_vel_d_armature/model.sdf" ;;
    esac
}
SETUPS="original original_armature corrected corrected_armature"
for setup in $SETUPS; do
    gazebo/record.sh 0 0 0 30 30 "q2/gz_${setup}_stand" "policy:=walk clock:=sim $(args_for $setup)" | grep -E "^Done|rror" || true
    gazebo/record.sh 0.5 0 0 24 4 "q2/gz_${setup}_walk" "policy:=walk clock:=sim $(args_for $setup)" | grep -E "^Done|rror" || true
done

Q=recordings/q2
for scenario in stand walk; do
    seconds=30; what="zero command for 30 s"
    [ "$scenario" = walk ] && seconds=24 && what="stand 4 s, then walk at 0.5 m/s"
    .venv/bin/python gazebo/side_by_side.py "$Q/gazebo_grid_$scenario.mp4" --seconds 120 --cols 2 \
        --title "walk.onnx in Gazebo, sim clock ($what)" \
        "1. Original controller|$Q/gz_original_$scenario.mp4|$Q/gz_original_$scenario.csv" \
        "2. Original + armature|$Q/gz_original_armature_$scenario.mp4|$Q/gz_original_armature_$scenario.csv" \
        "3. Corrected controller|$Q/gz_corrected_$scenario.mp4|$Q/gz_corrected_$scenario.csv" \
        "4. Corrected + armature|$Q/gz_corrected_armature_$scenario.mp4|$Q/gz_corrected_armature_$scenario.csv"
done
