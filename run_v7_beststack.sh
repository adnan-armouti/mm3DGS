#!/usr/bin/env bash
# 6-scene best-stack bench: C2b λ=1.0 + target_n=90000 + refined v_ego.
# See md/diagnostics/RESULTS.md §9 for the per-lever ablation that
# motivated this configuration.
#
# Expected wall-clock: 3 scenes per GPU × ~17 min/scene ≈ 51 min.
set -u
cd /home/adnan/Desktop/mm3DGS
PY=/home/adnan/.conda/envs/mmir/bin/python
export PYTHONUNBUFFERED=1

LOG=/home/adnan/Desktop/mm3DGS/logs_v7/beststack_6scene
mkdir -p "$LOG"

TRAIN_LOOPS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"

run_one() {
    local scene="$1" F="$2" gpu="$3"
    local TRAIN="$((F-4)),$((F-3)),$((F-2)),$((F-1)),$((F+1)),$((F+2)),$((F+3)),$((F+4))"
    local tag="${scene}__beststack"
    echo "[gpu $gpu] $tag starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v7.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$TRAIN" \
        --train_loops "$TRAIN_LOOPS" \
        --iters 500 --loss_type mse_raw --target_n 90000 \
        --doppler --loss_norm mean \
        --loss_multitask_lambda 1.0 \
        --use_refined_v_ego \
        > "$LOG/${tag}.log" 2>&1
    echo "[gpu $gpu] $tag done (exit=$?)"
}

GPU0=( "seq_0_frame_135:135" "seq_1_frame_185:185" "seq_1_frame_438:438" )
GPU1=( "seq_2_frame_105:105" "seq_2_frame_160:160" "seq_2_frame_300:300" )

run_gpu() {
    local gpu="$1"; shift
    for entry in "$@"; do
        IFS=':' read -r scene F <<< "$entry"
        run_one "$scene" "$F" "$gpu"
    done
}

( run_gpu 0 "${GPU0[@]}" ) & PID0=$!
( run_gpu 1 "${GPU1[@]}" ) & PID1=$!

wait $PID0; wait $PID1
echo "all done"
