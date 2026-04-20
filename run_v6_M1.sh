#!/usr/bin/env bash
# v6 M1 benchmark — 7-scene HO_8, diagnostic only.
# Pass criterion: final_test_cc matches v5 within MC noise (±0.003 per
# scene). Logs the new diag_normalised_gram_cc_test as a baseline reading.
set -u
cd /home/adnan/Desktop/mm3DGS
PY=/home/adnan/.conda/envs/mmir/bin/python
export PYTHONUNBUFFERED=1

LOG=/home/adnan/Desktop/mm3DGS/logs_v6_M1
mkdir -p "$LOG"

train_frames_for() {
    local F="$1"
    echo "$((F-4)),$((F-3)),$((F-2)),$((F-1)),$((F+1)),$((F+2)),$((F+3)),$((F+4))"
}

run_scene() {
    local scene="$1" F="$2" gpu="$3"
    local TRAIN=$(train_frames_for "$F")
    local tag="${scene}_M1"
    echo "[gpu $gpu] $tag starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v6.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$TRAIN" \
        --train_loops 0 \
        --iters 500 --loss_type mse_raw --target_n 20000 \
        --v6_milestone M1 \
        > "$LOG/${tag}.log" 2>&1
    echo "[gpu $gpu] $tag done"
}

# 4 scenes on GPU 0, 3 scenes on GPU 1 — mirrors v5 7-scene layout.
(
  run_scene seq_0_frame_135 135 0
  run_scene seq_0_frame_390 390 0
  run_scene seq_1_frame_185 185 0
  run_scene seq_1_frame_438 438 0
) &
PID0=$!
(
  run_scene seq_2_frame_105 105 1
  run_scene seq_2_frame_160 160 1
  run_scene seq_2_frame_300 300 1
) &
PID1=$!
wait $PID0; wait $PID1
echo "all done"
