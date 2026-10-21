#!/usr/bin/env bash
# Q1b, Gazebo check (run on the Mac): MuJoCo predicts that with implicit joint damping, Luis's gains and limits
# are harmless. Runs Unitree's policy headless on the luis_gains_vel_d variant (Luis's kp, kd, limits; D term as
# SDF joint damping): standing (zero command, 30 s) and walking (stand 4 s, then vx = 0.5, 24 s).
# Output: recordings/q1b/gazebo_<variant>_<scenario>_<n>.csv
# Usage: open_questions/q1b_controller/gazebo_confirm.sh [REPEATS] [VARIANT]
set -euo pipefail
REPEATS=${1:-2}; VARIANT=${2:-luis_gains_vel_d}
cd "$(dirname "$0")/../.."
.venv/bin/python gazebo/make_gain_variants.py > /dev/null
docker compose -f docker/compose.yaml up -d 2>/dev/null
MODEL=/home/ubuntu/humanoid-walking-lab/recordings/gz_models/$VARIANT/model.sdf
run() {   # VX SECONDS STAND NAME
    docker compose -f docker/compose.yaml exec -T -u ubuntu ros bash \
        /home/ubuntu/humanoid-walking-lab/gazebo/run_scenario.sh "$1" 0 0 "$2" "$3" "q1b/$4" \
        "policy:=unitree clock:=wall robot_model:=$MODEL" | grep "real-time"
}
for n in $(seq "$REPEATS"); do
    run 0 30 30 "gazebo_${VARIANT}_stand_$n"
    run 0.5 24 4 "gazebo_${VARIANT}_walk_$n"
done
