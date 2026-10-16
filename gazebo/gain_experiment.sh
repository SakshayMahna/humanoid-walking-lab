#!/usr/bin/env bash
# Runs INSIDE the container. Does the walking drift come from Gazebo's joint controllers?
# Same forward test (stand 4 s, then vx = 0.5 for 20 s) with three controller setups:
#   luis      Diego's g1_29dof model as-is (Luis's gains, ankle roll limited to 10 N·m)
#   mj        policy gains from the ONNX metadata / MuJoCo (make_gain_variants.py)
#   mj_vel_d  mj, with the D term on joint velocity instead of on the error
# First, on the Mac: .venv/bin/python gazebo/make_gain_variants.py  (writes recordings/gz_models/)
# Usage: gain_experiment.sh [REPEATS] [CLOCK]   ->  recordings/gains/<variant>_<policy>_<clock>_<n>.csv
set -eo pipefail
REPEATS=${1:-2}; CLOCK=${2:-wall}
LAB=~/humanoid-walking-lab
source /opt/ros/jazzy/setup.bash
(cd ~/ros2_ws && colcon build --packages-select g1_policy_runner > /dev/null)
mkdir -p $LAB/recordings/gains

declare -A MODEL=(
    [luis]=g1_free_roam
    [mj]=$LAB/recordings/gz_models/mj_gains/model.sdf
    [mj_vel_d]=$LAB/recordings/gz_models/mj_gains_vel_d/model.sdf
)
for n in $(seq "$REPEATS"); do
    for variant in luis mj mj_vel_d; do
        for policy in walk unitree; do
            name=gains/${variant}_${policy}_${CLOCK}_$n
            bash $LAB/gazebo/run_scenario.sh 0.5 0 0 24 4 "$name" \
                "policy:=$policy clock:=$CLOCK robot_model:=${MODEL[$variant]}" | tail -1
        done
    done
done
