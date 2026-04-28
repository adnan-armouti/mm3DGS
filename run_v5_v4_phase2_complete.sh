#!/usr/bin/env bash
# Complete Phase 2 grid: A1, A3, A4 across remaining 5 scenes (we already
# have A2, A5, baseline on all 6 and all 6 variants on seq_0_frame_135).
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
    if [ -f "${LOG}/${tag}.log" ] && grep -q "\[final\]" "${LOG}/${tag}.log" 2>/dev/null; then
        echo "[gpu $gpu] $tag SKIP (already done)"; return
    fi
    echo "[gpu $gpu] $tag starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v5_v4.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$TRAIN" \
        --train_loops 0 --iters 500 --loss_type mse_raw --target_n 20000 \
        --init_variant "$variant" \
        > "$LOG/${tag}.log" 2>&1
    echo "[gpu $gpu $variant] $tag done (exit=$?)"
}

# 5 scenes × 3 variants = 15 runs; split 8 + 7 across GPUs
GPU0_JOBS=(
  "A1_no_fps:seq_1_frame_185:185"  "A1_no_fps:seq_1_frame_438:438"
  "A1_no_fps:seq_2_frame_105:105"  "A1_no_fps:seq_2_frame_160:160"
  "A1_no_fps:seq_2_frame_300:300"
  "A3_union_amplitude:seq_1_frame_185:185"  "A3_union_amplitude:seq_1_frame_438:438"
  "A3_union_amplitude:seq_2_frame_105:105"
)
GPU1_JOBS=(
  "A3_union_amplitude:seq_2_frame_160:160"  "A3_union_amplitude:seq_2_frame_300:300"
  "A4_union_amp_lidar:seq_1_frame_185:185"  "A4_union_amp_lidar:seq_1_frame_438:438"
  "A4_union_amp_lidar:seq_2_frame_105:105"  "A4_union_amp_lidar:seq_2_frame_160:160"
  "A4_union_amp_lidar:seq_2_frame_300:300"
)

run_jobs() {
    local gpu="$1"; shift
    for j in "$@"; do
        IFS=':' read -r v scene F <<< "$j"
        run_one "$scene" "$F" "$v" "$gpu"
    done
}

# Also copy A1/A3/A4 result we already have for seq_0_frame_135 (from the ablation run)
for v in A1_no_fps A3_union_amplitude A4_union_amp_lidar; do
    if [ ! -f "$LOG/seq_0_frame_135__${v}.log" ]; then
        cp "/home/adnan/Desktop/mm3DGS/logs_v5_v4/init_ablation/seq_0_frame_135__${v}.log" \
           "$LOG/" 2>/dev/null || true
    fi
done

( run_jobs 0 "${GPU0_JOBS[@]}" ) & PID0=$!
( run_jobs 1 "${GPU1_JOBS[@]}" ) & PID1=$!
wait $PID0; wait $PID1
echo "all done"
