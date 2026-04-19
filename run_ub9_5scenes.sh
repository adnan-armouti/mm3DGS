#!/usr/bin/env bash
# UB_9 on 5 new scenes (seq_1_438 + seq_2_105 already done).
# 1 chirp × 9 frames = 9 train samples; ~1-2 min each.
set -u
cd /home/adnan/Desktop/mm3DGS
PY=/home/adnan/.conda/envs/mmir/bin/python
export PYTHONUNBUFFERED=1
LOG=/home/adnan/Desktop/mm3DGS/logs_ub9; mkdir -p "$LOG"

ub9_frames() { local F="$1"; echo "$((F-4)),$((F-3)),$((F-2)),$((F-1)),${F},$((F+1)),$((F+2)),$((F+3)),$((F+4))"; }

run_ub9() {
    local scene="$1" F="$2" gpu="$3"
    local TRAIN=$(ub9_frames "$F")
    echo "[gpu $gpu] $scene UB_9 starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v5.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$TRAIN" \
        --train_loops 0 \
        --iters 500 --loss_type mse_raw --target_n 20000 \
        > "$LOG/${scene}_UB9.log" 2>&1
    echo "[gpu $gpu] $scene UB_9 done"
}

(
  run_ub9 seq_0_frame_135 135 0
  run_ub9 seq_0_frame_390 390 0
  run_ub9 seq_2_frame_300 300 0
) &
PID0=$!
(
  run_ub9 seq_1_frame_185 185 1
  run_ub9 seq_2_frame_160 160 1
) &
PID1=$!
wait $PID0; wait $PID1
echo "all done"
