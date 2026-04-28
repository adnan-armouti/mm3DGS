#!/usr/bin/env bash
# v5_v4 Phase 3 with pool_knn child source (NEW Gaussians spawned from
# the post-visibility 850k LiDAR pool around high-‖∇p L‖ parents),
# instead of jitter (which adds Gaussian noise). This is the variant
# the user asked about: "selecting new points from the 850k pool".
#
# Two configurations:
#   (a) pool_knn + pos_grad_amp           (no learnable positions)
#   (b) pool_knn + pos_grad_amp + Phase 1 (combined; learnable amp-only)
set -u
cd /home/adnan/Desktop/mm3DGS
PY=/home/adnan/.conda/envs/mmir/bin/python
export PYTHONUNBUFFERED=1

LOG=/home/adnan/Desktop/mm3DGS/logs_v5_v4/pool_knn_6scene
mkdir -p "$LOG"

run_one() {
    local scene="$1" F="$2" config="$3" gpu="$4"
    local TRAIN="$((F-4)),$((F-3)),$((F-2)),$((F-1)),$((F+1)),$((F+2)),$((F+3)),$((F+4))"
    local tag="${scene}__${config}"
    echo "[gpu $gpu] $tag starting"

    local extra_flags=""
    if [ "$config" = "pool_combo" ]; then
        extra_flags="--learn_positions_lr 1e-5 --learn_positions_l2 100"
    fi

    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v5_v4.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$TRAIN" \
        --train_loops 0 --iters 500 --loss_type mse_raw --target_n 20000 \
        --reg_densify_interval 100 --reg_densify_split_frac 0.05 \
        --reg_densify_prune_frac 0.05 --reg_warm_iters 100 \
        --reg_densify_until 400 --reg_densify_child_source pool_knn \
        --reg_densify_pool_radius_m 0.15 \
        --densify_signal pos_grad_amp \
        $extra_flags \
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

# GPU 0 = pool_alone (no learnable positions); GPU 1 = pool_combo (with)
( run_gpu_config 0 "pool_alone" ) & PID0=$!
( run_gpu_config 1 "pool_combo" ) & PID1=$!
wait $PID0; wait $PID1
echo "all done"
