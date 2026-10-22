#!/usr/bin/env bash
# Q2 Test 3 (run on the Mac): headless repeats of Test 2, with the velocity input logged at full rate.
#   walk.onnx:  original, original_armature, corrected, corrected_armature
#   Unitree:    corrected, corrected_armature  (to compare the two policies with both fixes)
# Each: standing (zero command, 30 s) and walking (stand 4 s, then vx = 0.5), sim clock.
# Every odometry message (400 Hz) = walk.onnx's base_lin_vel input goes to <name>.odom.csv.
# Output: recordings/q2/rep_<policy>_<setup>_<scenario>_<n>.{csv,odom.csv,stats.log}
# Usage: open_questions/q2_walk_onnx/gazebo_repeats.sh [REPEATS]      (~50 s per run)
set -euo pipefail
REPEATS=${1:-3}
cd "$(dirname "$0")/../.."
.venv/bin/python gazebo/make_gain_variants.py > /dev/null
docker compose -f docker/compose.yaml up -d 2>/dev/null
L=/home/ubuntu/humanoid-walking-lab/recordings/gz_models
args_for() {   # a function: macOS bash 3.2 has no associative arrays
    case "$1" in
        original)           echo "robot_model:=g1_free_roam" ;;
        original_armature)  echo "robot_model:=$L/original_armature/model.sdf" ;;
        corrected)          echo "robot_model:=$L/mj_gains_vel_d/model.sdf" ;;
        corrected_armature) echo "robot_model:=$L/mj_gains_vel_d_armature/model.sdf" ;;
    esac
}
run() {   # POLICY SETUP VX SECONDS STAND NAME
    echo -n "$6: "
    docker compose -f docker/compose.yaml exec -T -u ubuntu -e ODOM_LOG=1 ros bash \
        /home/ubuntu/humanoid-walking-lab/gazebo/run_scenario.sh "$3" 0 0 "$4" "$5" "q2/$6" \
        "policy:=$1 clock:=sim $(args_for "$2")" | grep "real-time"
}
for n in $(seq "$REPEATS"); do
    for pair in walk:original walk:original_armature walk:corrected walk:corrected_armature \
                unitree:corrected unitree:corrected_armature; do
        policy=${pair%%:*}; setup=${pair#*:}
        run "$policy" "$setup" 0 30 30 "rep_${policy}_${setup}_stand_$n"
        run "$policy" "$setup" 0.5 24 4 "rep_${policy}_${setup}_walk_$n"
    done
done
