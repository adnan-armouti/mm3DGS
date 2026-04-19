#!/usr/bin/env bash
# S4 sweep: adaptive density (split top-Fisher + prune bottom-Fisher).
# Test split_frac ∈ {0.02, 0.05, 0.10} × interval ∈ {25, 50, 100}
# on seq_1 HO_8 (fast). 9 combos, ~2.5 min each, 2 GPUs → ~12 min.
set -u
cd /home/adnan/Desktop/mm3DGS
PY=/home/adnan/.conda/envs/mmir/bin/python
export PYTHONUNBUFFERED=1

LOG=/home/adnan/Desktop/mm3DGS/logs_s4
mkdir -p "$LOG"

SCENE=seq_1_frame_438
F=438
TRAIN=434,435,436,437,439,440,441,442

run_one() {
    local frac="$1" interval="$2" until_="$3" gpu="$4"
    local tag="s1H8_f${frac}_i${interval}_u${until_}"
    echo "[gpu $gpu] $tag starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v5.train_frame_nvs \
        --scene "$SCENE" --test_frame "$F" --train_frames "$TRAIN" \
        --train_loops 0 \
        --iters 500 --loss_type mse_raw --target_n 20000 \
        --reg_warm_iters 50 \
        --reg_densify_interval "$interval" \
        --reg_densify_split_frac "$frac" \
        --reg_densify_prune_frac "$frac" \
        --reg_densify_until "$until_" \
        --reg_densify_pos_jitter_m 0.02 \
        --reg_densify_mat_jitter 0.1 \
        > "$LOG/${tag}.log" 2>&1
    echo "[gpu $gpu] $tag done"
}

# GPU 0: smaller fractions + shorter intervals (more rounds, smaller moves)
(
  run_one 0.02 50  300 0
  run_one 0.02 25  300 0
  run_one 0.05 50  300 0
  run_one 0.05 100 300 0
) &
PID0=$!

# GPU 1: larger fractions + larger positions jitter variants
(
  run_one 0.10 50  300 1
  run_one 0.10 100 300 1
  run_one 0.02 100 400 1  # more rounds, later stop
  run_one 0.05 25  300 1
) &
PID1=$!

wait $PID0; wait $PID1
echo "all done"
