#!/usr/bin/env bash
# S2 λ=0.1 on HO_128 (both scenes) — tests whether the rotation-drift
# fix closes the gap in the regime where Analysis I's swap-ablation
# signal is strongest (+0.174 cc on seq_1, +0.048 on seq_2).
set -u
cd /home/adnan/Desktop/mm3DGS
PY=/home/adnan/.conda/envs/mmir/bin/python
export PYTHONUNBUFFERED=1

LAM=0.1
LOG=/home/adnan/Desktop/mm3DGS/logs_s2
mkdir -p "$LOG"

run_one() {
    local scene="$1" F="$2" train="$3" gpu="$4" tagbase="$5"
    echo "[gpu $gpu] $scene HO_128 \u03bb=${LAM} starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v5.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$train" \
        --iters 500 --loss_type mse_raw --target_n 20000 \
        --reg_fisher_rot_lambda "$LAM" --reg_warm_iters 50 \
        > "$LOG/${tagbase}_HO128_lam${LAM}.log" 2>&1
    echo "[gpu $gpu] $scene HO_128 done"
}

run_one seq_1_frame_438 438 434,435,436,437,439,440,441,442 0 seq_1 &
PID0=$!
run_one seq_2_frame_105 105 101,102,103,104,106,107,108,109 1 seq_2 &
PID1=$!

wait $PID0
wait $PID1
echo "all done"
