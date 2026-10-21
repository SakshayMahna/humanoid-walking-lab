#!/usr/bin/env bash
# Runs INSIDE the container (called by record.sh). Launches a policy stack, waits for the robot,
# sends a constant /cmd_vel, and screen-records the Gazebo window.
#
# Usage: record_inside.sh VX VY WZ SECONDS STAND_SECONDS NAME [POLICY_ARGS]
#   no POLICY_ARGS: Diego's README command (g1_locomotion, walk_free_elbows.onnx)
#   POLICY_ARGS:    our g1_policy_runner, e.g. "policy:=unitree clock:=sim"
set -eo pipefail

VX=$1; VY=$2; WZ=$3; SECONDS_TOTAL=$4; STAND=$5; NAME=$6; POLICY_ARGS=$7
FPS=${FPS:-30}   # screen-capture rate
LAB=~/humanoid-walking-lab
OUT=$LAB/recordings
mkdir -p "$OUT"
export DISPLAY=:1

source /opt/ros/jazzy/setup.bash

# Build the packages once (build/ and install/ live inside the container)
if [ ! -f ~/ros2_ws/install/g1_policy_runner/share/g1_policy_runner/package.xml ]; then
    (cd ~/ros2_ws && colcon build --packages-select g1_sim g1_locomotion g1_policy_runner)
fi
source ~/ros2_ws/install/setup.bash

# Stop anything left from a previous run (a leftover bridge would forward every message twice)
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

# Recording layout for the Gazebo GUI (only the 3D view + stats); Diego's code is untouched
mkdir -p ~/.gz/sim/8
cp $LAB/gazebo/gui_record.config ~/.gz/sim/8/gui.config

# Both start the inference node first, then (2 s later) Gazebo + bridge + spawn.
if [ -z "$POLICY_ARGS" ]; then
    ros2 launch g1_locomotion g1_locomotion.launch.py > "$OUT/$NAME.launch.log" 2>&1 &   # Diego's README
else
    # shellcheck disable=SC2086  (POLICY_ARGS is several launch arguments)
    ros2 launch g1_policy_runner policy.launch.py $POLICY_ARGS > "$OUT/$NAME.launch.log" 2>&1 &
fi
LAUNCH_PID=$!

# Wait until the robot exists and its joint states reach ROS
timeout 60 bash -c 'until ros2 topic echo --once /g1/joint_states > /dev/null 2>&1; do sleep 1; done'
# Wait for the GUI window; its scene needs a few seconds before /gui/follow takes effect
timeout 30 bash -c 'until xwininfo -root -tree | grep -q "\"Gazebo Sim\""; do sleep 1; done'
sleep 5

# Camera follows the robot from behind-left, slightly above
gz service -s /gui/follow --reqtype gz.msgs.StringMsg --reptype gz.msgs.Boolean \
    --timeout 3000 --req 'data: "g1"' > /dev/null
gz service -s /gui/follow/offset --reqtype gz.msgs.Vector3d --reptype gz.msgs.Boolean \
    --timeout 3000 --req 'x: -2.5, y: -2.5, z: 1.2' > /dev/null

# Find the Gazebo window on the virtual screen (absolute position = last geometry on the line)
GEOM=$(xwininfo -root -tree | grep '"Gazebo Sim"' | head -1 | grep -oE '[0-9]+x[0-9]+\+[0-9]+\+[0-9]+|\+[0-9]+\+[0-9]+$')
SIZE=$(echo "$GEOM" | head -1 | cut -d+ -f1)
X=$(echo "$GEOM" | tail -1 | cut -d+ -f2); Y=$(echo "$GEOM" | tail -1 | cut -d+ -f3)
echo "Recording Gazebo window $SIZE at +$X+$Y for ${SECONDS_TOTAL}s"

# Simulation time vs wall time, to compute the average real-time factor afterwards
# (read sec and nsec by name: protobuf omits fields that are 0, e.g. sec during the first second)
sim_seconds() { timeout 3 gz topic -e -t /stats -n 1 | awk '/^sim_time/{f=1; next} f && /^}/{exit}
                f && $1 == "sec:"{s = $2} f && $1 == "nsec:"{n = $2} END{printf "%.3f", s + n / 1e9}'; }
SIM0=$(sim_seconds); WALL0=$(date +%s.%N)

# -draw_mouse 0: hide the cursor; crop the GUI's 48 px title bar
ffmpeg -loglevel error -y -f x11grab -draw_mouse 0 -framerate "$FPS" -video_size "$SIZE" -i ":1+$X,$Y" \
    -t "$((SECONDS_TOTAL + 4))" -vf "crop=iw:ih-48:0:48" -c:v libx264 -preset ultrafast -pix_fmt yuv420p \
    "$OUT/$NAME.mp4" &
FFMPEG_PID=$!

# Stand still first, then walk with the requested command
# Command on simulation time (zero for STAND s, then the command) + odometry log, as in run_scenario.sh
echo "Command vx=$VX vy=$VY wz=$WZ after ${STAND}s (sim time)"
python3 $LAB/gazebo/scenario_logger.py "$OUT/$NAME.csv" "$SECONDS_TOTAL" "$STAND" "$VX" "$VY" "$WZ" || true

wait $FFMPEG_PID
SIM1=$(sim_seconds); WALL1=$(date +%s.%N)
POS=$(timeout 5 ros2 topic echo --once --field pose.pose.position /g1/pelvis/odometry | tr '\n' ' ')
{
    echo "command: vx=$VX vy=$VY wz=$WZ (after ${STAND}s standing)"
    echo "sim time: $SIM0 -> $SIM1 s, wall time: $(echo "$WALL1 - $WALL0" | bc) s"
    echo "average real-time factor: $(echo "scale=3; ($SIM1 - $SIM0) / ($WALL1 - $WALL0)" | bc)"
    echo "final pelvis position (odometry): $POS"
} | tee "$OUT/$NAME.stats.log"

kill $LAUNCH_PID 2>/dev/null || true
stop_stack
echo "Saved $OUT/$NAME.mp4"
