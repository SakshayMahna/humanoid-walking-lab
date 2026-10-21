#!/usr/bin/env bash
# Q1, test 1 (run on the Mac): is the drift while standing a start-up transient or sustained?
# Zero command for SECONDS, recorded, for two controller setups x two policies x REPEATS.
#   original   Diego's g1_29dof model as-is: original controller (Luis's gains, D on the error)
#   corrected  policy gains + D on joint velocity (recordings/gz_models/mj_gains_vel_d)
# Output: recordings/standing/stand_<setup>_<policy>_<n>.{mp4,gif,csv}
# Usage: [POLICIES="walk unitree"] open_questions/q1_standing/standing_test.sh [REPEATS] [SECONDS]
set -euo pipefail
REPEATS=${1:-3}; SECONDS_TOTAL=${2:-30}; POLICIES=${POLICIES:-walk unitree}
cd "$(dirname "$0")/../.."
M=/home/ubuntu/humanoid-walking-lab/recordings/gz_models
[ -f recordings/gz_models/mj_gains_vel_d/model.sdf ] || .venv/bin/python gazebo/make_gain_variants.py

for n in $(seq "$REPEATS"); do
    for setup in original corrected; do
        model=g1_free_roam
        [ "$setup" = corrected ] && model=$M/mj_gains_vel_d/model.sdf
        for policy in $POLICIES; do
            name=standing/stand_${setup}_${policy}_$n
            # STAND = SECONDS: the command stays zero for the whole run
            gazebo/record.sh 0 0 0 "$SECONDS_TOTAL" "$SECONDS_TOTAL" "$name" \
                "policy:=$policy clock:=wall robot_model:=$model" | grep -E "^Done|rror" || true
        done
    done
done
