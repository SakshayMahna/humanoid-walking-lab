#!/usr/bin/env bash
# Gazebo repeats (run on the Mac): Unitree's gains + joint damping, with and without armature, 3 standing runs (30 s)
# and 1 walking run (0.5 m/s) per policy. Existing recordings are kept. Output: recordings/training/rep/gz_{noarm,arm}_<label>_{stand,walk}_r<i>.*
# HEADLESS=1: no GUI and no video (gazebo/run_scenario.sh, as in the Week 3 runs), output in recordings/training/rep_headless/
set -uo pipefail
cd "$(dirname "$0")/.."
# (in a sandbox without access to ~/.docker, set BUILDX_CONFIG to a writable folder)
.venv/bin/python gazebo/make_gain_variants.py > /dev/null
R=/home/ubuntu/humanoid-walking-lab
G=training/kaggle/output
DIR=rep; [ "${HEADLESS:-0}" = 1 ] && DIR=rep_headless && docker compose -f docker/compose.yaml up -d 2>/dev/null
mkdir -p recordings/training/$DIR
scenario() {   # VX SECONDS STAND NAME ARGS
  if [ "${HEADLESS:-0}" = 1 ]; then
    docker compose -f docker/compose.yaml exec -T -u ubuntu ros bash $R/gazebo/run_scenario.sh "$1" 0 0 "$2" "$3" "$4" "$5" \
      | grep -E "rror" || true
  else
    gazebo/record.sh "$1" 0 0 "$2" "$3" "$4" "$5" | grep -E "rror" || true
  fi
}
policies=(
  "unitree|"
  "baseline_s1|$G/v6/logs/rsl_rl/g1_velocity/2026-10-26_03-05-28_baseline_seed1"
  "baseline_s2|$G/v15/logs/rsl_rl/g1_velocity/2026-10-27_10-29-22_baseline_seed2"
  "loo_foot_gait|$G/v9/logs/rsl_rl/g1_velocity/2026-10-26_09-24-37_loo_foot_gait_seed1"
  "dr_arm_height_s1|$G/v18/logs/rsl_rl/g1_velocity/2026-10-27_18-47-24_dr_armature_height_seed1"
  "dr_arm_height_s2|$G/v18/logs/rsl_rl/g1_velocity/2026-10-27_18-47-23_dr_armature_height_seed2"
  "ladder_0_strict|$G/v19/logs/rsl_rl/g1_velocity/2026-10-28_09-53-30_ladder_0_strict_seed1"
  "baseline_strict|$G/v19/logs/rsl_rl/g1_velocity/2026-10-28_09-53-30_baseline_strict_seed1"
  "track_hard|$G/v20/logs/rsl_rl/g1_velocity/2026-10-28_11-49-18_track_hard_seed1"
  "loo_fg_track_hard|$G/v20/logs/rsl_rl/g1_velocity/2026-10-28_11-49-18_loo_foot_gait_track_hard_seed1"
)
for variant in mj_gains_vel_d mj_gains_vel_d_armature; do
  arm=noarm; [ $variant = mj_gains_vel_d_armature ] && arm=arm
  for entry in "${policies[@]}"; do
    label=${entry%%|*}; run=${entry#*|}
    extra=""; [ -n "$run" ] && extra="policy_file:=$R/$run/policy.onnx"
    args="policy:=unitree clock:=sim robot_model:=$R/recordings/gz_models/$variant/model.sdf $extra"
    for i in 1 2 3; do
      out=recordings/training/$DIR/gz_${arm}_${label}_stand_r$i
      [ -f $out.csv ] || scenario 0 30 30 "${out#recordings/}" "$args"
    done
    out=recordings/training/$DIR/gz_${arm}_${label}_walk_r1
    [ -f $out.csv ] || scenario 0.5 24 4 "${out#recordings/}" "$args"
    echo "$(date +%H:%M) done $arm $label"
  done
done
