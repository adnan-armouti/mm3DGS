#!/usr/bin/env bash
# v7 Doppler production bench — 6 scenes, F±4 HO_8 training window,
# all 16 chirps per train frame via analytic Doppler phase modulation,
# 5.4×-fused step5_doppler CUDA kernel, post-C1 loss_norm=mean,
# refined v_ego pipeline ON.
#
# 6-scene mean from this configuration (md/diagnostics/RESULTS.md §11):
#   |RA|  train = 0.684    |RA|  test = 0.590
#   |RAD| train = 0.610    |RAD| test = 0.400
# Wall-clock: ~5–6 min/scene at target_n=20000 on a 4090.
#
# Override loss_norm for the legacy path via: LOSS_NORM=max ./run_v7_bench.sh
set -u
cd /home/adnan/Desktop/mm3DGS
PY=/home/adnan/.conda/envs/mmir/bin/python
export PYTHONUNBUFFERED=1

LOSS_NORM="${LOSS_NORM:-mean}"   # C1 default; override to 'max' for legacy

LOG=/home/adnan/Desktop/mm3DGS/logs_v7/norm_${LOSS_NORM}
mkdir -p "$LOG"

TRAIN_LOOPS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"

# NOTE: --use_refined_v_ego is the trainer's default. To opt out of
# refined v_ego (legacy seed cache), add --no-use_refined_v_ego below
# and drop the `_vegorf` implicit suffix in the output dir tagging.

run_one() {
    local scene="$1" F="$2" gpu="$3"
    local TRAIN="$((F-4)),$((F-3)),$((F-2)),$((F-1)),$((F+1)),$((F+2)),$((F+3)),$((F+4))"
    local tag="${scene}__v7doppler_norm${LOSS_NORM}"
    echo "[gpu $gpu] $tag starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v7.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$TRAIN" \
        --train_loops "$TRAIN_LOOPS" \
        --iters 500 --loss_type mse_raw --target_n 20000 \
        --doppler --loss_norm "$LOSS_NORM" \
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

( run_gpu 0 "${GPU0[@]}" ) &
PID0=$!
( run_gpu 1 "${GPU1[@]}" ) &
PID1=$!

wait $PID0; wait $PID1
echo "all done"
