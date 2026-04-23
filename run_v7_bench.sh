#!/usr/bin/env bash
# v7 Doppler bench — 6 scenes, data/ tree, F±4 HO_8 training window,
# all 16 chirps per train frame via analytic Doppler phase modulation
# (no 16× physical re-render). v5 mse_raw loss on the 3D |RAD| cube.
#
# See md/mm25dgs_v7_doppler_plan.md for the plan.
set -u
cd /home/adnan/Desktop/mm3DGS
PY=/home/adnan/.conda/envs/mmir/bin/python
export PYTHONUNBUFFERED=1

LOG=/home/adnan/Desktop/mm3DGS/logs_v7
mkdir -p "$LOG"

TRAIN_LOOPS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"

run_one() {
    local scene="$1" F="$2" gpu="$3"
    local TRAIN="$((F-4)),$((F-3)),$((F-2)),$((F-1)),$((F+1)),$((F+2)),$((F+3)),$((F+4))"
    local tag="${scene}__v7doppler"
    echo "[gpu $gpu] $tag starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v7.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$TRAIN" \
        --train_loops "$TRAIN_LOOPS" \
        --iters 500 --loss_type mse_raw --target_n 20000 \
        --doppler \
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
