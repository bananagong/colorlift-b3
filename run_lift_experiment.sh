#!/usr/bin/env bash
set -euo pipefail
source /home/catlab/mujoco_robosuite/activate.sh
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MPLBACKEND=Agg
experiment=/home/catlab/mujoco_robosuite/experiments/lift_act_v1
python -u -m lift_act.verify_env
python -u -m lift_act.collect --gpu 1 --count 50 --val-count 10
for count in 20 50; do
    if [[ ! -f "$experiment/act_$count/completed.json" ]]; then
        resume=()
        if [[ -f "$experiment/act_$count/last.pt" ]]; then resume=(--resume); fi
        python -u -m lift_act.train --demos "$count" --steps 6000 "${resume[@]}"
    fi
done
for count in 20 50; do
    if [[ ! -f "$experiment/act_$count/eval_test_q5/metrics.json" ]]; then
        python -u -m lift_act.evaluate --demos "$count" --gpu 1 --split test --episodes 20 --query-every 5
    fi
done
python -u -m lift_act.report
