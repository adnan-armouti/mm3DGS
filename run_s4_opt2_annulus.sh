#!/usr/bin/env bash
# Option 2 — pool_knn with random_annulus selection (sample uniformly in
# [r_min, r_max] instead of nearest). Uses the best-jitter turnover (0.02/50)
# for apples-to-apples comparison. 3 annulus configs × 7 scenes = 21 runs.
# ~2.5 min each, 2 GPUs parallel → ~30 min wallclock.
set -u
cd /home/adnan/Desktop/mm3DGS
PY=/home/adnan/.conda/envs/mmir/bin/python
export PYTHONUNBUFFERED=1

LOG=/home/adnan/Desktop/mm3DGS/logs_s4_opt2
mkdir -p "$LOG"

train_frames_for() {
    local F="$1"
    echo "$((F-4)),$((F-3)),$((F-2)),$((F-1)),$((F+1)),$((F+2)),$((F+3)),$((F+4))"
}

# Args: scene F gpu r_min r_max name
run_ann() {
    local scene="$1" F="$2" gpu="$3" rmin="$4" rmax="$5" name="$6"
    local TRAIN=$(train_frames_for "$F")
    local tag="${scene}_HO8_${name}"
    echo "[gpu $gpu] $tag starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v5.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$TRAIN" \
        --train_loops 0 \
        --iters 500 --loss_type mse_raw --target_n 20000 \
        --reg_warm_iters 50 \
        --reg_densify_interval 50 \
        --reg_densify_split_frac 0.02 \
        --reg_densify_prune_frac 0.02 \
        --reg_densify_until 300 \
        --reg_densify_child_source pool_knn \
        --reg_densify_pool_selection random_annulus \
        --reg_densify_pool_radius_min_m "$rmin" \
        --reg_densify_pool_radius_m "$rmax" \
        > "$LOG/${tag}.log" 2>&1
    echo "[gpu $gpu] $tag done"
}

# 3 annulus configs
# A: [0.02, 0.10]   — matches jitter σ=0.02m scale (2cm min, 10cm max)
# B: [0.02, 0.15]   — broader upper
# C: [0.05, 0.20]   — genuinely different points (5cm min, 20cm max)
CFGS=(
  "0.02 0.10 annA"
  "0.02 0.15 annB"
  "0.05 0.20 annC"
)

SCENES_G0=("seq_0_frame_135 135" "seq_0_frame_390 390" "seq_1_frame_185 185" "seq_1_frame_438 438")
SCENES_G1=("seq_2_frame_105 105" "seq_2_frame_160 160" "seq_2_frame_300 300")

(
  for sf in "${SCENES_G0[@]}"; do
    read scene F <<< "$sf"
    for cfg in "${CFGS[@]}"; do
      read rmin rmax name <<< "$cfg"
      run_ann "$scene" "$F" 0 "$rmin" "$rmax" "$name"
    done
  done
) &
PID0=$!

(
  for sf in "${SCENES_G1[@]}"; do
    read scene F <<< "$sf"
    for cfg in "${CFGS[@]}"; do
      read rmin rmax name <<< "$cfg"
      run_ann "$scene" "$F" 1 "$rmin" "$rmax" "$name"
    done
  done
) &
PID1=$!

wait $PID0; wait $PID1
echo "all done"
