#!/usr/bin/env bash
# v5_v4 Phase 2 — winning A5_amp_lidar_fps variant on all 6 scenes,
# v5-style single-chirp 8-train + 1-test held-out bench.
# Also re-runs baseline on the same 6 scenes (NEW v5_v4 baseline run,
# matched seed/MC noise) for apples-to-apples comparison.
set -u
cd /home/adnan/Desktop/mm3DGS
PY=/home/adnan/.conda/envs/mmir/bin/python
export PYTHONUNBUFFERED=1

LOG=/home/adnan/Desktop/mm3DGS/logs_v5_v4/a5_6scene
mkdir -p "$LOG"

run_one() {
    local scene="$1" F="$2" variant="$3" gpu="$4"
    local TRAIN="$((F-4)),$((F-3)),$((F-2)),$((F-1)),$((F+1)),$((F+2)),$((F+3)),$((F+4))"
    local tag="${scene}__${variant}"
    echo "[gpu $gpu] $tag starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v5_v4.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$TRAIN" \
        --train_loops 0 --iters 500 --loss_type mse_raw --target_n 20000 \
        --init_variant "$variant" \
        > "$LOG/${tag}.log" 2>&1
    echo "[gpu $gpu $variant] $tag done (exit=$?)"
}

# 12 jobs (6 scenes × 2 variants); 6 per GPU
SCENES=( "seq_0_frame_135:135" "seq_1_frame_185:185" "seq_1_frame_438:438"
         "seq_2_frame_105:105" "seq_2_frame_160:160" "seq_2_frame_300:300" )

run_gpu_var() {
    local gpu="$1" variant="$2"
    for entry in "${SCENES[@]}"; do
        IFS=':' read -r scene F <<< "$entry"
        run_one "$scene" "$F" "$variant" "$gpu"
    done
}

# GPU 0 = A5, GPU 1 = baseline
( run_gpu_var 0 "A5_amp_lidar_fps" ) & PID0=$!
( run_gpu_var 1 "baseline"          ) & PID1=$!

wait $PID0; wait $PID1
echo "all done"
