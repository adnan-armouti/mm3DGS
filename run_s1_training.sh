#!/usr/bin/env bash
# Stage 1 rerun: 4 variants × 2 scenes on matched position grids
# (seed_frame=test_frame default). target_n=20000. One scene per GPU.
set -u
cd /home/adnan/Desktop/mm3DGS
PY=/home/adnan/.conda/envs/mmir/bin/python
export PYTHONUNBUFFERED=1

LOG=/home/adnan/Desktop/mm3DGS/logs_s1
mkdir -p "$LOG"

run_scene_on_gpu() {
    local scene="$1" F="$2" gpu="$3"
    # HO: 8 train frames = F-4..F-1, F+1..F+4 (exclude test F)
    local HO_TRAIN="$((F-4)),$((F-3)),$((F-2)),$((F-1)),$((F+1)),$((F+2)),$((F+3)),$((F+4))"
    # UB: 9 train frames = F-4..F+4 (include test F)
    local UB_TRAIN="$((F-4)),$((F-3)),$((F-2)),$((F-1)),${F},$((F+1)),$((F+2)),$((F+3)),$((F+4))"

    # Run 4 variants sequentially on this GPU
    local t0
    t0=$(date +%s)

    # HO_128 (16 chirps × 8 frames = 128 train samples)
    echo "[gpu $gpu] $scene F=$F HO_128 starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v5.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$HO_TRAIN" \
        --iters 500 --loss_type mse_raw --target_n 20000 \
        > "$LOG/${scene}_HO_128.log" 2>&1
    echo "[gpu $gpu] $scene HO_128 done at $(($(date +%s)-t0))s"

    # HO_8 (1 chirp × 8 frames = 8 train samples, first chirp only)
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v5.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$HO_TRAIN" \
        --train_loops 0 \
        --iters 500 --loss_type mse_raw --target_n 20000 \
        > "$LOG/${scene}_HO_8.log" 2>&1
    echo "[gpu $gpu] $scene HO_8  done at $(($(date +%s)-t0))s"

    # UB_144 (16 chirps × 9 frames = 144 train samples)
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v5.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$UB_TRAIN" \
        --iters 500 --loss_type mse_raw --target_n 20000 \
        > "$LOG/${scene}_UB_144.log" 2>&1
    echo "[gpu $gpu] $scene UB_144 done at $(($(date +%s)-t0))s"

    # UB_9 (1 chirp × 9 frames = 9 train samples, first chirp only)
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v5.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$UB_TRAIN" \
        --train_loops 0 \
        --iters 500 --loss_type mse_raw --target_n 20000 \
        > "$LOG/${scene}_UB_9.log" 2>&1
    echo "[gpu $gpu] $scene UB_9  done at $(($(date +%s)-t0))s"
}

# Scene 1 on GPU 0, scene 2 on GPU 1 — parallel
run_scene_on_gpu seq_1_frame_438 438 0 &
PID0=$!
run_scene_on_gpu seq_2_frame_105 105 1 &
PID1=$!

wait $PID0
wait $PID1
echo "all done"
