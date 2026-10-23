#!/usr/bin/env bash
# Runs INSIDE the container. Headless run (no Gazebo GUI) that logs pelvis odometry to CSV.
# Same order as Diego's launch file: inference node first, Gazebo + bridge + spawn 2 s later.
#
# Usage: run_scenario.sh VX VY WZ SECONDS STAND_SECONDS NAME [POLICY_ARGS]   ->  recordings/NAME.csv
#   no POLICY_ARGS: Diego's g1_locomotion node (walk_free_elbows.onnx)
#   POLICY_ARGS:    our g1_policy_runner, e.g. "policy:=unitree clock:=sim"
# ODOM_LOG=1: also log every odometry message (400 Hz) to recordings/NAME.odom.csv
# STEP_LOG=1: also log the wall and sim time of every policy step to recordings/NAME.steps.csv
set -eo pipefail

VX=$1; VY=$2; WZ=$3; SECONDS_TOTAL=$4; STAND=$5; NAME=$6; POLICY_ARGS=$7
LAB=~/humanoid-walking-lab
mkdir -p "$(dirname "$LAB/recordings/$NAME")"
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

# Simulation time from Gazebo's /stats, to compute the average real-time factor of the run
# (read sec and nsec by name: protobuf omits fields that are 0, e.g. sec during the first second)
sim_seconds() { timeout 3 gz topic -e -t /stats -n 1 | awk '/^sim_time/{f=1; next} f && /^}/{exit}
                f && $1 == "sec:"{s = $2} f && $1 == "nsec:"{n = $2} END{printf "%.3f", s + n / 1e9}'; }
SIM0=$(sim_seconds); WALL0=$(date +%s.%N)

# The logger also publishes /cmd_vel, timed on simulation time (zero for STAND s, then the command)
LOG_ARGS=""
[ "${ODOM_LOG:-0}" = 1 ] && LOG_ARGS="--odom $LAB/recordings/$NAME.odom.csv"
[ "${STEP_LOG:-0}" = 1 ] && LOG_ARGS="$LOG_ARGS --steps $LAB/recordings/$NAME.steps.csv"
# shellcheck disable=SC2086  (LOG_ARGS is empty or option pairs)
python3 $LAB/gazebo/scenario_logger.py $LAB/recordings/$NAME.csv "$SECONDS_TOTAL" "$STAND" "$VX" "$VY" "$WZ" $LOG_ARGS || true

SIM1=$(sim_seconds); WALL1=$(date +%s.%N)
{
    echo "sim time: $SIM0 -> $SIM1 s, wall time: $(echo "$WALL1 - $WALL0" | bc) s"
    echo "average real-time factor: $(echo "scale=3; ($SIM1 - $SIM0) / ($WALL1 - $WALL0)" | bc)"
} | tee $LAB/recordings/$NAME.stats.log
stop_stack
echo "Saved $LAB/recordings/$NAME.csv"
