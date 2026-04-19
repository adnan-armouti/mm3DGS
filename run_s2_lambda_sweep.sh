#!/usr/bin/env bash
# S2 λ sweep: Fisher-weighted rotation drift, seq_1 HO_8 variant.
# 5 values × ~2.5 min each; parallel 2 GPUs → ~8 min wallclock.
set -u
cd /home/adnan/Desktop/mm3DGS
PY=/home/adnan/.conda/envs/mmir/bin/python
export PYTHONUNBUFFERED=1

SCENE=seq_1_frame_438
F=438
TRAIN=434,435,436,437,439,440,441,442

LOG=/home/adnan/Desktop/mm3DGS/logs_s2
mkdir -p "$LOG"

run_one() {
    local lam="$1" gpu="$2"
    local tag="lam${lam}"
    echo "[gpu $gpu] $tag starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v5.train_frame_nvs \
        --scene "$SCENE" --test_frame "$F" --train_frames "$TRAIN" \
        --train_loops 0 \
        --iters 500 --loss_type mse_raw --target_n 20000 \
        --reg_fisher_rot_lambda "$lam" --reg_warm_iters 50 \
        > "$LOG/seq_1_HO8_${tag}.log" 2>&1
    echo "[gpu $gpu] $tag done"
}

# GPU 0: λ ∈ {1e-3, 1e-2, 1e-1} sequential
(
  run_one 1e-3 0
  run_one 1e-2 0
  run_one 1e-1 0
) &
PID0=$!

# GPU 1: λ ∈ {1, 10} sequential
(
  run_one 1 1
  run_one 10 1
) &
PID1=$!

wait $PID0
wait $PID1
echo "all done"
