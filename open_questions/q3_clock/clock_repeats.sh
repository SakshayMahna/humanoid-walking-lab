#!/usr/bin/env bash
# Q3 (run on the Mac): does the policy timer's clock (wall vs simulation) change the result?
# Corrected controller, with and without armature, both policies, both clocks, standing and walking, headless.
# Every policy step's wall and sim time is logged (STEP_LOG=1) to check how regular the 20 ms period really is.
# Output: recordings/q3/<policy>_<setup>_<clock>_<scenario>_<n>.{csv,steps.csv,stats.log}
# Usage: open_questions/q3_clock/clock_repeats.sh [REPEATS]      (~50 s per run, 16 runs per repeat)
set -euo pipefail
REPEATS=${1:-3}
cd "$(dirname "$0")/../.."
.venv/bin/python gazebo/make_gain_variants.py > /dev/null
docker compose -f docker/compose.yaml up -d 2>/dev/null
L=/home/ubuntu/humanoid-walking-lab/recordings/gz_models
model_for() {
    case "$1" in
        corrected)          echo "$L/mj_gains_vel_d/model.sdf" ;;
        corrected_armature) echo "$L/mj_gains_vel_d_armature/model.sdf" ;;
    esac
}
run() {   # POLICY SETUP CLOCK VX SECONDS STAND NAME
    echo -n "$7: "
    docker compose -f docker/compose.yaml exec -T -u ubuntu -e STEP_LOG=1 ros bash \
        /home/ubuntu/humanoid-walking-lab/gazebo/run_scenario.sh "$4" 0 0 "$5" "$6" "q3/$7" \
        "policy:=$1 clock:=$3 robot_model:=$(model_for "$2")" | grep "real-time"
}
for n in $(seq "$REPEATS"); do
    for policy in unitree walk; do
        for setup in corrected corrected_armature; do
            for clock in wall sim; do
                run "$policy" "$setup" "$clock" 0 30 30 "${policy}_${setup}_${clock}_stand_$n"
                run "$policy" "$setup" "$clock" 0.5 24 4 "${policy}_${setup}_${clock}_walk_$n"
            done
        done
    done
done
