#!/usr/bin/env bash
# S4 validation: winner config (0.02 frac × interval 50 × until 300)
# on seq_2 HO_8 (fast validate) + seq_1/seq_2 HO_128 (Analysis I regime).
set -u
cd /home/adnan/Desktop/mm3DGS
PY=/home/adnan/.conda/envs/mmir/bin/python
export PYTHONUNBUFFERED=1

LOG=/home/adnan/Desktop/mm3DGS/logs_s4
mkdir -p "$LOG"

run_one() {
    local scene="$1" F="$2" train="$3" loops="$4" tag="$5" gpu="$6" frac="$7" intv="$8" untl="$9"
    echo "[gpu $gpu] $tag starting"
    local LOOP_ARG=""
    if [[ -n "$loops" ]]; then LOOP_ARG="--train_loops $loops"; fi
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v5.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$train" \
        $LOOP_ARG \
        --iters 500 --loss_type mse_raw --target_n 20000 \
        --reg_warm_iters 50 \
        --reg_densify_interval "$intv" \
        --reg_densify_split_frac "$frac" \
        --reg_densify_prune_frac "$frac" \
        --reg_densify_until "$untl" \
        --reg_densify_pos_jitter_m 0.02 \
        --reg_densify_mat_jitter 0.1 \
        > "$LOG/${tag}.log" 2>&1
    echo "[gpu $gpu] $tag done"
}

S1=seq_1_frame_438; F1=438; T1=434,435,436,437,439,440,441,442
S2=seq_2_frame_105; F2=105; T2=101,102,103,104,106,107,108,109

# GPU 0: seq_2 HO_8 validation (fast) → then seq_1 HO_128 (long)
(
  run_one $S2 $F2 $T2 0 "s2_HO8_S4_winner"  0 0.02 50  300
  run_one $S1 $F1 $T1 "" "s1_HO128_S4_winner" 0 0.02 50  300
) &
PID0=$!

# GPU 1: seq_2 HO_8 runner-up → then seq_2 HO_128 (long)
(
  run_one $S2 $F2 $T2 0 "s2_HO8_S4_alt"     1 0.10 100 300
  run_one $S2 $F2 $T2 "" "s2_HO128_S4_winner" 1 0.02 50  300
) &
PID1=$!

wait $PID0; wait $PID1
echo "all done"
