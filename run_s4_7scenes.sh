#!/usr/bin/env bash
# 7-scene HO_8 benchmark: matched-grid baseline + S4 pool_knn (big pool).
# 5 new baselines (seq_1_438 + seq_2_105 already exist) + 7 S4 runs = 12 new runs.
# Config: split_frac=0.02, interval=50, until=300, pool_radius=0.15
# ~2.5 min per run × 6 runs per GPU = ~15 min wallclock on 2 GPUs.
set -u
cd /home/adnan/Desktop/mm3DGS
PY=/home/adnan/.conda/envs/mmir/bin/python
export PYTHONUNBUFFERED=1

LOG=/home/adnan/Desktop/mm3DGS/logs_s4_7scenes
mkdir -p "$LOG"

# HO_8 frame list for a given centre F: F-4..F-1, F+1..F+4
train_frames_for() {
    local F="$1"
    echo "$((F-4)),$((F-3)),$((F-2)),$((F-1)),$((F+1)),$((F+2)),$((F+3)),$((F+4))"
}

run_baseline() {
    local scene="$1" F="$2" gpu="$3"
    local TRAIN=$(train_frames_for "$F")
    local tag="${scene}_HO8_base"
    echo "[gpu $gpu] $tag starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v5.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$TRAIN" \
        --train_loops 0 \
        --iters 500 --loss_type mse_raw --target_n 20000 \
        > "$LOG/${tag}.log" 2>&1
    echo "[gpu $gpu] $tag done"
}

run_s4_pk() {
    local scene="$1" F="$2" gpu="$3"
    local TRAIN=$(train_frames_for "$F")
    local tag="${scene}_HO8_S4pk"
    echo "[gpu $gpu] $tag starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v5.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$TRAIN" \
        --train_loops 0 \
        --iters 500 --loss_type mse_raw --target_n 20000 \
        --reg_warm_iters 50 \
        --reg_densify_interval 50 \
        --reg_densify_split_frac 0.02 \
        --reg_densify_prune_frac 0.02 \
        --reg_densify_until 300 \
        --reg_densify_child_source pool_knn \
        --reg_densify_pool_radius_m 0.15 \
        > "$LOG/${tag}.log" 2>&1
    echo "[gpu $gpu] $tag done"
}

# GPU 0 (6 runs): seq_0_135, seq_0_390, seq_1_185 — baseline + S4 each
(
  run_baseline seq_0_frame_135 135 0
  run_s4_pk    seq_0_frame_135 135 0
  run_baseline seq_0_frame_390 390 0
  run_s4_pk    seq_0_frame_390 390 0
  run_baseline seq_1_frame_185 185 0
  run_s4_pk    seq_1_frame_185 185 0
) &
PID0=$!

# GPU 1 (6 runs): seq_2_160, seq_2_300 baseline+S4; seq_1_438 + seq_2_105 S4 only
(
  run_baseline seq_2_frame_160 160 1
  run_s4_pk    seq_2_frame_160 160 1
  run_baseline seq_2_frame_300 300 1
  run_s4_pk    seq_2_frame_300 300 1
  run_s4_pk    seq_1_frame_438 438 1
  run_s4_pk    seq_2_frame_105 105 1
) &
PID1=$!

wait $PID0; wait $PID1
echo "all done"
