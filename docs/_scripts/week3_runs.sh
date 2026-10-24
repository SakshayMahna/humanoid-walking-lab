#!/usr/bin/env bash
# Gazebo runs for the Week 3 post that were missing on the sim clock, so every table in the post compares runs
# made the same way (sim clock, headless, standing 30 s / walking: stand 4 s then vx = 0.5).
# Already available on the sim clock: walk.onnx original / corrected / corrected + armature and Unitree corrected /
# corrected + armature (recordings/q2/rep_*), Unitree original + armature / 0.2 ms step (recordings/q1b/gz_*).
# Output: recordings/week3/<policy>_<variant>_<scenario>_<n>.csv
# Usage (run on the Mac): docs/_scripts/week3_runs.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
.venv/bin/python gazebo/make_gain_variants.py > /dev/null
docker compose -f docker/compose.yaml up -d 2>/dev/null
L=/home/ubuntu/humanoid-walking-lab/recordings/gz_models
model_for() {
    case "$1" in
        original)         echo "g1_free_roam" ;;
        mj_gains)         echo "$L/mj_gains/model.sdf" ;;
        luis_gains_vel_d) echo "$L/luis_gains_vel_d/model.sdf" ;;
    esac
}
run() {   # POLICY VARIANT N
    for scenario in stand walk; do
        vx=0; seconds=30; stand=30
        [ "$scenario" = walk ] && vx=0.5 && seconds=24 && stand=4
        name="$1_$2_${scenario}_$3"
        # skip runs that already have a complete log (rerunning the script resumes)
        if [ -f "recordings/week3/$name.csv" ] && [ "$(wc -l < "recordings/week3/$name.csv")" -gt 400 ]; then continue; fi
        echo -n "$name: "
        docker compose -f docker/compose.yaml exec -T -u ubuntu ros bash \
            /home/ubuntu/humanoid-walking-lab/gazebo/run_scenario.sh "$vx" 0 0 "$seconds" "$stand" \
            "week3/$name" "policy:=$1 clock:=sim robot_model:=$(model_for "$2")" | grep "real-time" || echo "no result"
    done
}
for n in 1 2 3; do
    run unitree original "$n"
    run unitree mj_gains "$n"
    run walk mj_gains "$n"
done
for n in 1 2; do
    run unitree luis_gains_vel_d "$n"
done
