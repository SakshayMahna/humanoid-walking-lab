#!/usr/bin/env bash
# Q1 test 2 (run on the Mac): does the corrected controller also stand still WITHOUT the Gazebo window?
# Same zero-command run as standing_test.sh, but headless (gui:=false, no recording), logged from the moment the
# robot's joint states appear, with the real-time factor recorded.
# Output: recordings/standing/headless_<setup>_<policy>_<n>.{csv,stats.log}
# Usage: [SETUPS="corrected"] [POLICIES="unitree"] open_questions/q1_standing/headless_test.sh [REPEATS] [SECONDS]
set -euo pipefail
REPEATS=${1:-3}; SECONDS_TOTAL=${2:-30}; SETUPS=${SETUPS:-corrected}; POLICIES=${POLICIES:-unitree}
cd "$(dirname "$0")/../.."
M=/home/ubuntu/humanoid-walking-lab/recordings/gz_models
[ -f recordings/gz_models/mj_gains_vel_d/model.sdf ] || .venv/bin/python gazebo/make_gain_variants.py
docker compose -f docker/compose.yaml up -d

for n in $(seq "$REPEATS"); do
    for setup in $SETUPS; do
        model=g1_free_roam
        [ "$setup" = corrected ] && model=$M/mj_gains_vel_d/model.sdf
        for policy in $POLICIES; do
            docker compose -f docker/compose.yaml exec -T -u ubuntu ros bash \
                /home/ubuntu/humanoid-walking-lab/gazebo/run_scenario.sh 0 0 0 "$SECONDS_TOTAL" "$SECONDS_TOTAL" \
                "standing/headless_${setup}_${policy}_$n" "policy:=$policy clock:=wall robot_model:=$model" \
                | grep -E "real-time"
        done
    done
done
