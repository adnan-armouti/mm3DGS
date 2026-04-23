#!/usr/bin/env bash
# v6 M2 — Doppler-extended v5-identical loss on |RAD| cube.
# 6 scenes, data/ preprocessing + alignment (CANONICAL BASELINE — we
# compare M2 directly against the v5/M1 HO_8 data/ table), F±4
# training window, all 16 chirps rendered per train frame. mse_raw
# loss on |RAD| (D=31, Az=127, R=256) after Hann + ifftshift + fft +
# drop-bin-0 + fftshift on both azimuth and chirp axes, N_DOP=32.
#
# Note: data/ only has aligned configs for F-4..F+4, so the 2 edge
# frames (F-4 and F+4) fall back to frame-own-pose across all 16
# chirps (no Doppler signature from those frames). 6/8 interior train
# frames retain proper per-chirp LERP + Doppler signal. Acceptable
# loss for this first-validation M2 pass.
#
# TDM velocity-induced phase compensation is DEFERRED — v5's renderer
# does not model it; revisit per md/doppler_forward_model_plan.md if
# the |RAD| loss diverges from the v5 |RA|-magnitude diagnostic.
set -u
cd /home/adnan/Desktop/mm3DGS
PY=/home/adnan/.conda/envs/mmir/bin/python
export PYTHONUNBUFFERED=1

LOG=/home/adnan/Desktop/mm3DGS/logs_v6_M2
mkdir -p "$LOG"

# All 16 chirps: 0..15
TRAIN_LOOPS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"

run_one() {
    local scene="$1" F="$2" gpu="$3"
    local TRAIN="$((F-4)),$((F-3)),$((F-2)),$((F-1)),$((F+1)),$((F+2)),$((F+3)),$((F+4))"
    local tag="${scene}__M2"
    echo "[gpu $gpu] $tag starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v6.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$TRAIN" \
        --train_loops "$TRAIN_LOOPS" \
        --iters 500 --loss_type mse_raw --target_n 20000 \
        --v6_milestone M2 \
        --save_ra_pngs \
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
