#!/usr/bin/env bash
# Option 1 — same-config S4 sweep across 7 scenes (HO_8).
# Tests 3 new configs against the already-run pool_knn (0.02/50/r=0.15).
#   B: jitter 0.02/50 (jitter winner on seq_1 in earlier 2-scene test)
#   C: jitter 0.10/100 (jitter alt, best cross-scene in earlier 2-scene test)
#   D: pool_knn 0.10/100/r=0.15 (alt pool_knn)
# 3 configs × 7 scenes = 21 runs, ~2.5 min each, 2 GPUs → ~30 min wallclock.
set -u
cd /home/adnan/Desktop/mm3DGS
PY=/home/adnan/.conda/envs/mmir/bin/python
export PYTHONUNBUFFERED=1

LOG=/home/adnan/Desktop/mm3DGS/logs_s4_opt1
mkdir -p "$LOG"

train_frames_for() {
    local F="$1"
    echo "$((F-4)),$((F-3)),$((F-2)),$((F-1)),$((F+1)),$((F+2)),$((F+3)),$((F+4))"
}

# Args: scene F gpu cfg_name frac intv until src {pos_jit|radius}
run_cfg() {
    local scene="$1" F="$2" gpu="$3" name="$4" frac="$5" intv="$6" untl="$7" src="$8" sig="$9"
    local TRAIN=$(train_frames_for "$F")
    local tag="${scene}_HO8_${name}"
    local EXTRA=""
    if [[ "$src" == "jitter" ]]; then
        EXTRA="--reg_densify_child_source jitter --reg_densify_pos_jitter_m $sig"
    else
        EXTRA="--reg_densify_child_source pool_knn --reg_densify_pool_radius_m $sig"
    fi
    echo "[gpu $gpu] $tag starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v5.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$TRAIN" \
        --train_loops 0 \
        --iters 500 --loss_type mse_raw --target_n 20000 \
        --reg_warm_iters 50 \
        --reg_densify_interval "$intv" \
        --reg_densify_split_frac "$frac" \
        --reg_densify_prune_frac "$frac" \
        --reg_densify_until "$untl" \
        $EXTRA \
        > "$LOG/${tag}.log" 2>&1
    echo "[gpu $gpu] $tag done"
}

# Scene list + frame indices
SCENES=("seq_0_frame_135 135" "seq_0_frame_390 390" "seq_1_frame_185 185" \
        "seq_1_frame_438 438" "seq_2_frame_105 105" "seq_2_frame_160 160" \
        "seq_2_frame_300 300")

# Configs: (name, frac, intv, until, src, sig)
# B = jitter 0.02/50 pos=0.02m
# C = jitter 0.10/100 pos=0.02m
# D = pool_knn 0.10/100 r=0.15
CFG_B="jit002i50 0.02 50 300 jitter 0.02"
CFG_C="jit010i100 0.10 100 300 jitter 0.02"
CFG_D="pk010i100 0.10 100 300 pool_knn 0.15"

# GPU 0 gets seq_0_135, seq_0_390, seq_1_185, seq_1_438 (4 scenes × 3 cfgs = 12 runs)
# GPU 1 gets seq_2_105, seq_2_160, seq_2_300 (3 scenes × 3 cfgs = 9 runs)
# Total: 21 runs; GPU 0 has 30 min sequential, GPU 1 has ~23 min.
(
  for sf in "seq_0_frame_135 135" "seq_0_frame_390 390" "seq_1_frame_185 185" "seq_1_frame_438 438"; do
    read scene F <<< "$sf"
    for cfg in "$CFG_B" "$CFG_C" "$CFG_D"; do
      read name frac intv untl src sig <<< "$cfg"
      run_cfg "$scene" "$F" 0 "$name" "$frac" "$intv" "$untl" "$src" "$sig"
    done
  done
) &
PID0=$!

(
  for sf in "seq_2_frame_105 105" "seq_2_frame_160 160" "seq_2_frame_300 300"; do
    read scene F <<< "$sf"
    for cfg in "$CFG_B" "$CFG_C" "$CFG_D"; do
      read name frac intv untl src sig <<< "$cfg"
      run_cfg "$scene" "$F" 1 "$name" "$frac" "$intv" "$untl" "$src" "$sig"
    done
  done
) &
PID1=$!

wait $PID0; wait $PID1
echo "all done"
