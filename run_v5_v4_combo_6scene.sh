#!/usr/bin/env bash
# v5_v4 Phase 3 + Phase 1 combined — best single-scene result so far
# (seq_0_frame_135: test_cc = 0.639 vs baseline 0.575).
# Configuration:
#   - baseline init
#   - learn_positions_lr=1e-5, L2 anchor=100 (sub-mm bounded amp-only)
#   - densify with pos_grad_amp signal, 5%/100iter, jitter children
set -u
cd /home/adnan/Desktop/mm3DGS
PY=/home/adnan/.conda/envs/mmir/bin/python
export PYTHONUNBUFFERED=1

LOG=/home/adnan/Desktop/mm3DGS/logs_v5_v4/combo_6scene
mkdir -p "$LOG"

run_one() {
    local scene="$1" F="$2" gpu="$3"
    local TRAIN="$((F-4)),$((F-3)),$((F-2)),$((F-1)),$((F+1)),$((F+2)),$((F+3)),$((F+4))"
    local tag="${scene}__combo"
    echo "[gpu $gpu] $tag starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v5_v4.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$TRAIN" \
        --train_loops 0 --iters 500 --loss_type mse_raw --target_n 20000 \
        --reg_densify_interval 100 --reg_densify_split_frac 0.05 \
        --reg_densify_prune_frac 0.05 --reg_warm_iters 100 \
        --reg_densify_until 400 --reg_densify_child_source jitter \
        --densify_signal pos_grad_amp \
        --learn_positions_lr 1e-5 --learn_positions_l2 100 \
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
