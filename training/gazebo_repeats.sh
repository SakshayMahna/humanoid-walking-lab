#!/usr/bin/env bash
# Gazebo repeats (run on the Mac): Unitree's gains + joint damping, with and without armature, 3 standing runs (30 s)
# and 1 walking run (0.5 m/s) per policy. Existing recordings are kept. Output: recordings/training/rep/gz_{noarm,arm}_<label>_{stand,walk}_r<i>.*
set -uo pipefail
cd "$(dirname "$0")/.."
# (in a sandbox without access to ~/.docker, set BUILDX_CONFIG to a writable folder)
.venv/bin/python gazebo/make_gain_variants.py > /dev/null
R=/home/ubuntu/humanoid-walking-lab
G=training/kaggle/output
mkdir -p recordings/training/rep
policies=(
  "unitree|"
  "baseline_s1|$G/v6/logs/rsl_rl/g1_velocity/2026-10-26_03-05-28_baseline_seed1"
  "baseline_s2|$G/v15/logs/rsl_rl/g1_velocity/2026-10-27_10-29-22_baseline_seed2"
  "loo_foot_gait|$G/v9/logs/rsl_rl/g1_velocity/2026-10-26_09-24-37_loo_foot_gait_seed1"
  "dr_arm_height_s1|$G/v18/logs/rsl_rl/g1_velocity/2026-10-27_18-47-24_dr_armature_height_seed1"
  "dr_arm_height_s2|$G/v18/logs/rsl_rl/g1_velocity/2026-10-27_18-47-23_dr_armature_height_seed2"
)
for variant in mj_gains_vel_d mj_gains_vel_d_armature; do
  arm=noarm; [ $variant = mj_gains_vel_d_armature ] && arm=arm
  for entry in "${policies[@]}"; do
    label=${entry%%|*}; run=${entry#*|}
    extra=""; [ -n "$run" ] && extra="policy_file:=$R/$run/policy.onnx"
    args="policy:=unitree clock:=sim robot_model:=$R/recordings/gz_models/$variant/model.sdf $extra"
    for i in 1 2 3; do
      out=recordings/training/rep/gz_${arm}_${label}_stand_r$i
      [ -f $out.csv ] || gazebo/record.sh 0 0 0 30 30 "${out#recordings/}" "$args" | grep -E "rror"
    done
    out=recordings/training/rep/gz_${arm}_${label}_walk_r1
    [ -f $out.csv ] || gazebo/record.sh 0.5 0 0 24 4 "${out#recordings/}" "$args" | grep -E "rror"
    echo "$(date +%H:%M) done $arm $label"
  done
done
