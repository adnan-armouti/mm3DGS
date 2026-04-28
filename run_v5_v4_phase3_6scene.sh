#!/usr/bin/env bash
# v5_v4 Phase 3 6-scene bench:
#   variant A (baseline-no-densify) — already in logs_v5_v4/a5_6scene/
#   variant B (densify, fisher signal)
#   variant C (densify, pos_grad_amp signal)
# All baseline init. 500 iter, target_n=20K, single-chirp v5-style.
set -u
cd /home/adnan/Desktop/mm3DGS
PY=/home/adnan/.conda/envs/mmir/bin/python
export PYTHONUNBUFFERED=1

LOG=/home/adnan/Desktop/mm3DGS/logs_v5_v4/phase3_6scene
mkdir -p "$LOG"

run_one() {
    local scene="$1" F="$2" signal="$3" gpu="$4"
    local TRAIN="$((F-4)),$((F-3)),$((F-2)),$((F-1)),$((F+1)),$((F+2)),$((F+3)),$((F+4))"
    local tag="${scene}__densify_${signal}"
    echo "[gpu $gpu] $tag starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v5_v4.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$TRAIN" \
        --train_loops 0 --iters 500 --loss_type mse_raw --target_n 20000 \
        --reg_densify_interval 100 --reg_densify_split_frac 0.05 \
        --reg_densify_prune_frac 0.05 --reg_warm_iters 100 \
        --reg_densify_until 400 --reg_densify_child_source jitter \
        --densify_signal "$signal" \
        > "$LOG/${tag}.log" 2>&1
    echo "[gpu $gpu] $tag done (exit=$?)"
}

SCENES=( "seq_0_frame_135:135" "seq_1_frame_185:185" "seq_1_frame_438:438"
         "seq_2_frame_105:105" "seq_2_frame_160:160" "seq_2_frame_300:300" )

run_gpu_signal() {
    local gpu="$1" signal="$2"
    for entry in "${SCENES[@]}"; do
        IFS=':' read -r scene F <<< "$entry"
        run_one "$scene" "$F" "$signal" "$gpu"
    done
}

# GPU 0 = pos_grad_amp; GPU 1 = fisher (legacy v5)
( run_gpu_signal 0 "pos_grad_amp" ) & PID0=$!
( run_gpu_signal 1 "fisher"        ) & PID1=$!

wait $PID0; wait $PID1
echo "all done"
