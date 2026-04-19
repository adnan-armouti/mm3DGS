#!/usr/bin/env bash
# S4 pool_knn comparison on HO_8 (both scenes).
# Compares child_source=pool_knn vs jitter with matched hyperparameters.
# 6 runs × ~2.5 min, parallel 2 GPUs → ~10 min wallclock.
set -u
cd /home/adnan/Desktop/mm3DGS
PY=/home/adnan/.conda/envs/mmir/bin/python
export PYTHONUNBUFFERED=1

LOG=/home/adnan/Desktop/mm3DGS/logs_s4_knn
mkdir -p "$LOG"

run_one() {
    local scene="$1" F="$2" train="$3" frac="$4" intv="$5" untl="$6" src="$7" radius="$8" gpu="$9" tag="${10}"
    echo "[gpu $gpu] $tag starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v5.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$train" \
        --train_loops 0 \
        --iters 500 --loss_type mse_raw --target_n 20000 \
        --reg_warm_iters 50 \
        --reg_densify_interval "$intv" \
        --reg_densify_split_frac "$frac" \
        --reg_densify_prune_frac "$frac" \
        --reg_densify_until "$untl" \
        --reg_densify_child_source "$src" \
        --reg_densify_pool_radius_m "$radius" \
        > "$LOG/${tag}.log" 2>&1
    echo "[gpu $gpu] $tag done"
}

S1=seq_1_frame_438; F1=438; T1=434,435,436,437,439,440,441,442
S2=seq_2_frame_105; F2=105; T2=101,102,103,104,106,107,108,109

# GPU 0: seq_1 HO_8 — pool_knn at winner config, 3 radii
(
  run_one $S1 $F1 $T1 0.02 50  300 pool_knn 0.10 0 "s1H8_pk_f0.02_i50_r0.10"
  run_one $S1 $F1 $T1 0.02 50  300 pool_knn 0.15 0 "s1H8_pk_f0.02_i50_r0.15"
  run_one $S1 $F1 $T1 0.02 50  300 pool_knn 0.30 0 "s1H8_pk_f0.02_i50_r0.30"
) &
PID0=$!

# GPU 1: seq_2 HO_8 — pool_knn at winner+alt cfg; seq_1 alt cfg
(
  run_one $S2 $F2 $T2 0.02 50  300 pool_knn 0.15 1 "s2H8_pk_f0.02_i50_r0.15"
  run_one $S2 $F2 $T2 0.10 100 300 pool_knn 0.15 1 "s2H8_pk_f0.10_i100_r0.15"
  run_one $S1 $F1 $T1 0.10 100 300 pool_knn 0.15 1 "s1H8_pk_f0.10_i100_r0.15"
) &
PID1=$!

wait $PID0; wait $PID1
echo "all done"
