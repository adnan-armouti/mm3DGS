#!/usr/bin/env bash
# v5_v4 Phase 2 — init-variant ablation on a single scene.
# Variants (per md/mm25dgs_v5_v4_smart_sampling_and_densification.md §2.5):
#   baseline           — v5 init (FOV+RX+seed-cosine+FPS)
#   A1_no_fps          — drops FPS
#   A2_union_cos       — union over train poses, cos θ_i only
#   A3_union_amplitude — union, cos θ_i · 1/d²
#   A4_union_amp_lidar — A3 × LiDAR intensity (production candidate)
#   A5_amp_lidar_fps   — A4 then mild FPS
#
# Wall-clock: 5 × ~5 min ≈ 25 min serial, ~13 min on 2 GPUs.
set -u
cd /home/adnan/Desktop/mm3DGS
PY=/home/adnan/.conda/envs/mmir/bin/python
export PYTHONUNBUFFERED=1

LOG=/home/adnan/Desktop/mm3DGS/logs_v5_v4/init_ablation
mkdir -p "$LOG"

SCENE=seq_0_frame_135
F=135
TRAIN="131,132,133,134,136,137,138,139"

run_one() {
    local variant="$1" gpu="$2"
    local tag="${SCENE}__${variant}"
    echo "[gpu $gpu] $tag starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v5_v4.train_frame_nvs \
        --scene "$SCENE" --test_frame "$F" --train_frames "$TRAIN" \
        --train_loops 0 --iters 500 --loss_type mse_raw --target_n 20000 \
        --init_variant "$variant" \
        > "$LOG/${tag}.log" 2>&1
    echo "[gpu $gpu] $tag done (exit=$?)"
}

# 6 variants split across 2 GPUs (3-3)
GPU0=( "baseline" "A1_no_fps" "A2_union_cos" )
GPU1=( "A3_union_amplitude" "A4_union_amp_lidar" "A5_amp_lidar_fps" )

run_gpu() {
    local gpu="$1"; shift
    for v in "$@"; do run_one "$v" "$gpu"; done
}

( run_gpu 0 "${GPU0[@]}" ) & PID0=$!
( run_gpu 1 "${GPU1[@]}" ) & PID1=$!

wait $PID0; wait $PID1
echo "all done"
