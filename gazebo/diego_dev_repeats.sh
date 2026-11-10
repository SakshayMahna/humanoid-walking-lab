#!/usr/bin/env bash
# Diego's newer node (policy_manager_main, development branch, walk.onnx) in Gazebo, headless:
# his robot model vs our corrected one (training gains, D as joint damping, armature in the links),
# at his policy rate (60 Hz) and at the training rate (50 Hz). 3 standing runs (30 s) and 1 walking run
# (stand 4 s, then 0.5 m/s) each. Existing results are kept.
# The rate is changed in the INSTALLED config inside the container (a copy made by colcon), never in his source.
# Usage (on the Mac, his clone on the development branch and built in the container): gazebo/diego_dev_repeats.sh
# Output: recordings/diego_dev/<model>_<hz>_{stand_r<i>,walk_r1}.csv
set -uo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python gazebo/make_gain_variants.py > /dev/null
COMPOSE="docker compose -f docker/compose.yaml"
$COMPOSE up -d 2>/dev/null
R=/home/ubuntu/humanoid-walking-lab
CFG=/home/ubuntu/ros2_ws/install/g1_locomotion/share/g1_locomotion/config/walk.yaml
set_rate() { $COMPOSE exec -T -u ubuntu ros sed -i "s/^frequency_hz: .*/frequency_hz: $1/" $CFG; }

for hz in 60 50; do
  set_rate "$hz.0"
  for model in his corrected; do
    sim=""; [ $model = corrected ] && sim="robot_model:=$R/recordings/gz_models/mj_gains_vel_d_armature/model.sdf bridge_config:=g1_pos"   # custom models need the bridge named
    run() {   # VX SECONDS STAND NAME
      [ -f recordings/diego_dev/$4.csv ] && return
      $COMPOSE exec -T -u ubuntu -e DIEGO_NODE=policy_manager_main -e SIM_ARGS="$sim" ros \
        bash $R/gazebo/run_scenario.sh "$1" 0 0 "$2" "$3" "diego_dev/$4" "" | grep -E "rror" || true
    }
    for i in 1 2 3; do run 0 30 30 ${model}_${hz}hz_stand_r$i; done
    run 0.5 24 4 ${model}_${hz}hz_walk_r1
    echo "$(date +%H:%M) done $model $hz Hz"
  done
done
set_rate 60.0     # back to his setting
