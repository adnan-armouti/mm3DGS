#!/usr/bin/env bash
# Single chained driver: Phase 2 grid retry → pool_knn 6-scene → Phase 4 6-scene
set -u
cd /home/adnan/Desktop/mm3DGS
PY=/home/adnan/.conda/envs/mmir/bin/python
export PYTHONUNBUFFERED=1

LOG_P2=/home/adnan/Desktop/mm3DGS/logs_v5_v4/a5_6scene
LOG_PK=/home/adnan/Desktop/mm3DGS/logs_v5_v4/pool_knn_6scene
LOG_P4=/home/adnan/Desktop/mm3DGS/logs_v5_v4/phase4_6scene
mkdir -p "$LOG_P2" "$LOG_PK" "$LOG_P4"

run_phase2() {
    local scene="$1" F="$2" variant="$3" gpu="$4"
    local TRAIN="$((F-4)),$((F-3)),$((F-2)),$((F-1)),$((F+1)),$((F+2)),$((F+3)),$((F+4))"
    local tag="${scene}__${variant}"
    if [ -f "${LOG_P2}/${tag}.log" ] && grep -q "\[final\]" "${LOG_P2}/${tag}.log" 2>/dev/null; then
        echo "[gpu $gpu] $tag SKIP"; return
    fi
    echo "[gpu $gpu] phase2 $tag starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v5_v4.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$TRAIN" \
        --train_loops 0 --iters 500 --loss_type mse_raw --target_n 20000 \
        --init_variant "$variant" \
        > "$LOG_P2/${tag}.log" 2>&1
    echo "[gpu $gpu] phase2 $tag done (exit=$?)"
}

run_poolknn() {
    local scene="$1" F="$2" config="$3" gpu="$4"
    local TRAIN="$((F-4)),$((F-3)),$((F-2)),$((F-1)),$((F+1)),$((F+2)),$((F+3)),$((F+4))"
    local tag="${scene}__${config}"
    local extra=""
    if [ "$config" = "pool_combo" ]; then
        extra="--learn_positions_lr 1e-5 --learn_positions_l2 100"
    fi
    echo "[gpu $gpu] poolknn $tag starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v5_v4.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$TRAIN" \
        --train_loops 0 --iters 500 --loss_type mse_raw --target_n 20000 \
        --reg_densify_interval 100 --reg_densify_split_frac 0.05 \
        --reg_densify_prune_frac 0.05 --reg_warm_iters 100 \
        --reg_densify_until 400 --reg_densify_child_source pool_knn \
        --reg_densify_pool_radius_m 0.15 --densify_signal pos_grad_amp \
        $extra > "$LOG_PK/${tag}.log" 2>&1
    echo "[gpu $gpu] poolknn $tag done (exit=$?)"
}

run_phase4() {
    local scene="$1" F="$2" config="$3" gpu="$4"
    local TRAIN="$((F-4)),$((F-3)),$((F-2)),$((F-1)),$((F+1)),$((F+2)),$((F+3)),$((F+4))"
    local tag="${scene}__${config}"
    local extra=""
    if [ "$config" = "p4_combo" ]; then
        extra="--reg_densify_interval 100 --reg_densify_split_frac 0.05 \
               --reg_densify_prune_frac 0.05 --reg_warm_iters 100 \
               --reg_densify_until 400 --reg_densify_child_source jitter \
               --densify_signal pos_grad_amp \
               --learn_positions_lr 1e-5 --learn_positions_l2 100"
    fi
    echo "[gpu $gpu] phase4 $tag starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v5_v4.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$TRAIN" \
        --train_loops 0 --iters 500 --loss_type mse_raw --target_n 20000 \
        --phase4_d_lambda 0.001 --phase4_d_test_knn 2 \
        $extra > "$LOG_P4/${tag}.log" 2>&1
    echo "[gpu $gpu] phase4 $tag done (exit=$?)"
}

SCENES=( "seq_0_frame_135:135" "seq_1_frame_185:185" "seq_1_frame_438:438"
         "seq_2_frame_105:105" "seq_2_frame_160:160" "seq_2_frame_300:300" )

# === Phase 2 grid retries (failed exit=1 entries) ===
echo "=== Phase 2 grid retries ==="
GPU0_P2=( "A1_no_fps:seq_2_frame_105:105" "A1_no_fps:seq_2_frame_160:160"
          "A1_no_fps:seq_2_frame_300:300" "A3_union_amplitude:seq_1_frame_185:185"
          "A3_union_amplitude:seq_1_frame_438:438" "A3_union_amplitude:seq_2_frame_105:105" )
GPU1_P2=( "A4_union_amp_lidar:seq_1_frame_185:185" "A4_union_amp_lidar:seq_1_frame_438:438"
          "A4_union_amp_lidar:seq_2_frame_105:105" "A4_union_amp_lidar:seq_2_frame_160:160"
          "A4_union_amp_lidar:seq_2_frame_300:300" )
( for j in "${GPU0_P2[@]}"; do IFS=':' read -r v s F <<< "$j"
    if [ -f "$LOG_P2/${s}__${v}.log" ] && grep -q "\[final\]" "$LOG_P2/${s}__${v}.log"; then
        rm "$LOG_P2/${s}__${v}.log" 2>/dev/null  # remove failed one
    fi
    run_phase2 "$s" "$F" "$v" 0; done ) & PG0=$!
( for j in "${GPU1_P2[@]}"; do IFS=':' read -r v s F <<< "$j"
    if [ -f "$LOG_P2/${s}__${v}.log" ] && grep -q "\[final\]" "$LOG_P2/${s}__${v}.log"; then
        rm "$LOG_P2/${s}__${v}.log" 2>/dev/null
    fi
    run_phase2 "$s" "$F" "$v" 1; done ) & PG1=$!
# Actually for failed runs, let me just delete the failed logs first
for j in "${GPU0_P2[@]}" "${GPU1_P2[@]}"; do
    IFS=':' read -r v s F <<< "$j"
    if [ -f "$LOG_P2/${s}__${v}.log" ] && ! grep -q "\[final\]" "$LOG_P2/${s}__${v}.log"; then
        rm -f "$LOG_P2/${s}__${v}.log"
    fi
done
wait $PG0; wait $PG1
echo "phase2 retries done"

# === pool_knn 6-scene ===
echo "=== pool_knn 6-scene ==="
( for entry in "${SCENES[@]}"; do IFS=':' read -r s F <<< "$entry"; run_poolknn "$s" "$F" "pool_alone" 0; done ) & P0=$!
( for entry in "${SCENES[@]}"; do IFS=':' read -r s F <<< "$entry"; run_poolknn "$s" "$F" "pool_combo" 1; done ) & P1=$!
wait $P0; wait $P1
echo "poolknn done"

# === Phase 4 6-scene ===
echo "=== phase4 6-scene ==="
( for entry in "${SCENES[@]}"; do IFS=':' read -r s F <<< "$entry"; run_phase4 "$s" "$F" "p4_alone" 0; done ) & P0=$!
( for entry in "${SCENES[@]}"; do IFS=':' read -r s F <<< "$entry"; run_phase4 "$s" "$F" "p4_combo" 1; done ) & P1=$!
wait $P0; wait $P1
echo "phase4 done"
echo "all done"
