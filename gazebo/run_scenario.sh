#!/usr/bin/env bash
# Runs INSIDE the container. Headless run (no Gazebo GUI) that logs pelvis odometry to CSV.
# Same order as Diego's launch file: inference node first, Gazebo + bridge + spawn 2 s later.
#
# Usage: run_scenario.sh VX VY WZ SECONDS STAND_SECONDS NAME [POLICY_ARGS]   ->  recordings/NAME.csv
#   no POLICY_ARGS: Diego's g1_locomotion node (walk_free_elbows.onnx)
#   POLICY_ARGS:    our g1_policy_runner, e.g. "policy:=unitree clock:=sim"
set -eo pipefail

VX=$1; VY=$2; WZ=$3; SECONDS_TOTAL=$4; STAND=$5; NAME=$6; POLICY_ARGS=$7
LAB=~/humanoid-walking-lab
mkdir -p $LAB/recordings
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

stop_stack() {
    pkill -f "ros2 launch" || true
    pkill -f "ros2 run" || true
    pkill -f parameter_bridge || true
    pkill -f "gz sim" || true
    pkill -f g1_locomotion_main || true
    pkill -f "lib/g1_policy_runner/policy_runner" || true
}
stop_stack
sleep 2

if [ -z "$POLICY_ARGS" ]; then
    ros2 run g1_locomotion g1_locomotion_main > $LAB/recordings/$NAME.inference.log 2>&1 &
    sleep 2
    ros2 launch "$(ros2 pkg prefix g1_sim)/share/g1_sim/g1_sim/sim_core.py" gui:=false \
        > $LAB/recordings/$NAME.sim.log 2>&1 &
else
    # shellcheck disable=SC2086  (POLICY_ARGS is several launch arguments)
    ros2 launch g1_policy_runner policy.launch.py gui:=false $POLICY_ARGS > $LAB/recordings/$NAME.launch.log 2>&1 &
fi

timeout 60 bash -c 'until ros2 topic echo --once /g1/joint_states > /dev/null 2>&1; do sleep 1; done'
# The logger also publishes /cmd_vel, timed on simulation time (zero for STAND s, then the command)
python3 $LAB/gazebo/scenario_logger.py $LAB/recordings/$NAME.csv "$SECONDS_TOTAL" "$STAND" "$VX" "$VY" "$WZ" || true
stop_stack
echo "Saved $LAB/recordings/$NAME.csv"
