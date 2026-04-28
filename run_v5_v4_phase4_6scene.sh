#!/usr/bin/env bash
# v5_v4 Phase 4 (D) — per-train-view membership matrix m ∈ R^{N × V} with
# sigmoid(m[:, v]) as opacity multiplier per train view. L1 sparsity loss
# encourages each point to be owned by few views. At test, KNN-average
# over the K nearest train views (default K=2).
#
# Two configs:
#   (a) phase4 alone (baseline init, no densify)
#   (b) phase4 + Phase 1 + Phase 3 jitter (the prior best combo)
set -u
cd /home/adnan/Desktop/mm3DGS
PY=/home/adnan/.conda/envs/mmir/bin/python
export PYTHONUNBUFFERED=1

LOG=/home/adnan/Desktop/mm3DGS/logs_v5_v4/phase4_6scene
mkdir -p "$LOG"

run_one() {
    local scene="$1" F="$2" config="$3" gpu="$4"
    local TRAIN="$((F-4)),$((F-3)),$((F-2)),$((F-1)),$((F+1)),$((F+2)),$((F+3)),$((F+4))"
    local tag="${scene}__${config}"
    echo "[gpu $gpu] $tag starting"

    local extra=""
    if [ "$config" = "p4_combo" ]; then
        extra="--reg_densify_interval 100 --reg_densify_split_frac 0.05 \
               --reg_densify_prune_frac 0.05 --reg_warm_iters 100 \
               --reg_densify_until 400 --reg_densify_child_source jitter \
               --densify_signal pos_grad_amp \
               --learn_positions_lr 1e-5 --learn_positions_l2 100"
    fi

    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v5_v4.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$TRAIN" \
        --train_loops 0 --iters 500 --loss_type mse_raw --target_n 20000 \
        --phase4_d_lambda 0.001 --phase4_d_test_knn 2 \
        $extra \
        > "$LOG/${tag}.log" 2>&1
    echo "[gpu $gpu] $tag done (exit=$?)"
}

SCENES=( "seq_0_frame_135:135" "seq_1_frame_185:185" "seq_1_frame_438:438"
         "seq_2_frame_105:105" "seq_2_frame_160:160" "seq_2_frame_300:300" )

run_gpu_config() {
    local gpu="$1" config="$2"
    for entry in "${SCENES[@]}"; do
        IFS=':' read -r scene F <<< "$entry"
        run_one "$scene" "$F" "$config" "$gpu"
    done
}

( run_gpu_config 0 "p4_alone" ) & PID0=$!
( run_gpu_config 1 "p4_combo" ) & PID1=$!
wait $PID0; wait $PID1
echo "all done"
