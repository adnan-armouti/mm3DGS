#!/usr/bin/env bash
# S2 refinement: tighter sweep on seq_1 + seq_2 HO_8 validation.
set -u
cd /home/adnan/Desktop/mm3DGS
PY=/home/adnan/.conda/envs/mmir/bin/python
export PYTHONUNBUFFERED=1

LOG=/home/adnan/Desktop/mm3DGS/logs_s2
mkdir -p "$LOG"

run_one() {
    local scene="$1" F="$2" train="$3" lam="$4" gpu="$5" tagbase="$6"
    echo "[gpu $gpu] $scene $tagbase $lam starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v5.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$train" \
        --train_loops 0 \
        --iters 500 --loss_type mse_raw --target_n 20000 \
        --reg_fisher_rot_lambda "$lam" --reg_warm_iters 50 \
        > "$LOG/${tagbase}_lam${lam}.log" 2>&1
    echo "[gpu $gpu] $scene $tagbase $lam done"
}

# GPU 0: seq_1 HO_8 tight sweep (λ ∈ {0.03, 0.3}) + HO_128 winner validation
(
  run_one seq_1_frame_438 438 434,435,436,437,439,440,441,442 0.03 0 seq_1_HO8
  run_one seq_1_frame_438 438 434,435,436,437,439,440,441,442 0.3  0 seq_1_HO8
) &
PID0=$!

# GPU 1: seq_2 HO_8 validation at λ ∈ {0.03, 0.1, 0.3}
(
  run_one seq_2_frame_105 105 101,102,103,104,106,107,108,109 0.03 1 seq_2_HO8
  run_one seq_2_frame_105 105 101,102,103,104,106,107,108,109 0.1  1 seq_2_HO8
  run_one seq_2_frame_105 105 101,102,103,104,106,107,108,109 0.3  1 seq_2_HO8
) &
PID1=$!

wait $PID0
wait $PID1
echo "all done"
