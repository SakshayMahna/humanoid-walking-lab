#!/usr/bin/env bash
# Q1b, Gazebo experiments A and B (run on the Mac). Both keep Diego's ORIGINAL controller (Luis's gains and
# limits, D in the PID: explicit, with the kick) and remove the instability another way:
#   A  original_armature  each motor's armature added to its link inertia (make_gain_variants.py)
#   B  step_0p2ms         physics step 0.2 ms instead of 1 ms: the explicit condition scales with the step
#   baseline              original, 1 ms
# All with clock:=sim, so the policy runs every 20 ms of simulated time even when Gazebo is slower than real time
# (B is ~5x more work). Unitree's policy; standing (zero command, 30 s) and walking (stand 4 s, then vx = 0.5).
# Output: recordings/q1b/gz_<setup>_<scenario>_<n>.{csv,stats.log}
# Usage: open_questions/q1b_controller/gazebo_armature_step.sh [REPEATS]
set -euo pipefail
REPEATS=${1:-2}
cd "$(dirname "$0")/../.."
.venv/bin/python gazebo/make_gain_variants.py > /dev/null

# World with a 0.2 ms step: Diego's empty world plus a <physics> block (it has none, so Gazebo uses 1 ms)
mkdir -p recordings/gz_models/worlds
python3 - <<'EOF'
src = open("3p/2025-tfg-diego-lopez/g1_sim/worlds/empty_world.sdf").read()
physics = ('    <physics name="0p2ms" type="ignored">\n      <max_step_size>0.0002</max_step_size>\n'
           '      <real_time_factor>1.0</real_time_factor>\n    </physics>\n')
i = src.index("<world name=")
i = src.index("\n", i) + 1
open("recordings/gz_models/worlds/empty_0p2ms.sdf", "w").write(src[:i] + physics + src[i:])
EOF

docker compose -f docker/compose.yaml up -d 2>/dev/null
L=/home/ubuntu/humanoid-walking-lab/recordings/gz_models
args_for() {   # launch arguments per setup (a function: macOS bash 3.2 has no associative arrays)
    case "$1" in
        baseline)   echo "robot_model:=g1_free_roam" ;;
        armature)   echo "robot_model:=$L/original_armature/model.sdf" ;;
        step_0p2ms) echo "robot_model:=g1_free_roam world:=$L/worlds/empty_0p2ms.sdf" ;;
    esac
}
run() {   # SETUP VX SECONDS STAND NAME
    echo -n "$5: "
    docker compose -f docker/compose.yaml exec -T -u ubuntu ros bash \
        /home/ubuntu/humanoid-walking-lab/gazebo/run_scenario.sh "$2" 0 0 "$3" "$4" "q1b/$5" \
        "policy:=unitree clock:=sim $(args_for "$1")" | grep "real-time"
}
for n in $(seq "$REPEATS"); do
    for setup in baseline armature step_0p2ms; do
        run "$setup" 0 30 30 "gz_${setup}_stand_$n"
        run "$setup" 0.5 24 4 "gz_${setup}_walk_$n"
    done
done
