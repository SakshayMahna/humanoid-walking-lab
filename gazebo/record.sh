#!/usr/bin/env bash
# Run Diego's G1 stack in Gazebo (Docker) and record it, without opening the VNC desktop.
#
# Usage (from the repo root):
#   gazebo/record.sh [VX] [VY] [WZ] [SECONDS] [STAND_SECONDS] [NAME] [POLICY_ARGS]
#   gazebo/record.sh 0.5 0 0 20 4 diego_forward                          # Diego's node as-is
#   gazebo/record.sh 0.5 0 0 20 4 unitree_forward "policy:=unitree clock:=sim"   # g1_policy_runner
#   FPS=30 GIF_FPS=25 gazebo/record.sh ...                                 # frame rates (defaults)
#
# Output in recordings/: NAME.mp4, NAME.gif, NAME.launch.log, NAME.stats.log (RTF + pelvis position)
set -euo pipefail

VX=${1:-0.5}; VY=${2:-0.0}; WZ=${3:-0.0}
SECONDS_TOTAL=${4:-20}; STAND=${5:-4}; NAME=${6:-walk}; POLICY_ARGS=${7:-}
FPS=${FPS:-30}; GIF_FPS=${GIF_FPS:-25}
cd "$(dirname "$0")/.."
COMPOSE="docker compose -f docker/compose.yaml"

$COMPOSE up -d --build
# Wait for the VNC X server (display :1) inside the container
$COMPOSE exec -T -u ubuntu ros bash -c 'until [ -e /tmp/.X11-unix/X1 ]; do sleep 1; done'

$COMPOSE exec -T -u ubuntu -e FPS="$FPS" ros bash /home/ubuntu/humanoid-walking-lab/gazebo/record_inside.sh \
    "$VX" "$VY" "$WZ" "$SECONDS_TOTAL" "$STAND" "$NAME" "$POLICY_ARGS"

# Small GIF next to the MP4 (made inside the container with its ffmpeg; written to the mounted repo)
$COMPOSE exec -T -u ubuntu ros bash -c "cd ~/humanoid-walking-lab/recordings && \
    ffmpeg -loglevel error -y -i $NAME.mp4 -vf 'fps=$GIF_FPS,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse' $NAME.gif"
echo "Done: recordings/$NAME.mp4 and recordings/$NAME.gif"
