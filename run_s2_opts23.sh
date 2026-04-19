#!/usr/bin/env bash
# S2 Options 2 (EMA target) + 3 (angle threshold) — sweep on HO_8 both scenes.
# Each run ~2.5 min at target_n=20k; 12 runs total ⇒ ~15 min wallclock on 2 GPUs.
set -u
cd /home/adnan/Desktop/mm3DGS
PY=/home/adnan/.conda/envs/mmir/bin/python
export PYTHONUNBUFFERED=1

LOG=/home/adnan/Desktop/mm3DGS/logs_s2_opts23
mkdir -p "$LOG"

run_one() {
    local scene="$1" F="$2" train="$3" lam="$4" target="$5" ema="$6" thr="$7" gpu="$8" tag="$9"
    echo "[gpu $gpu] $scene $tag starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v5.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$train" \
        --train_loops 0 \
        --iters 500 --loss_type mse_raw --target_n 20000 \
        --reg_fisher_rot_lambda "$lam" --reg_warm_iters 50 \
        --reg_fisher_rot_target "$target" \
        --reg_fisher_rot_ema_alpha "$ema" \
        --reg_fisher_rot_threshold_deg "$thr" \
        > "$LOG/${tag}.log" 2>&1
    echo "[gpu $gpu] $scene $tag done"
}

# GPU 0: seq_1 HO_8 — Option 2 (EMA target) sweep over λ, and Option 3 (thresh).
S1=seq_1_frame_438; F1=438; T1=434,435,436,437,439,440,441,442
S2=seq_2_frame_105; F2=105; T2=101,102,103,104,106,107,108,109

(
  # Option 2: EMA target, fixed α=0.9. Sweep λ ∈ {0.1, 1, 10} (EMA is a
  # softer target than init so larger λ may work).
  run_one $S1 $F1 $T1 0.1  ema  0.9  0    0 "s1H8_o2_ema09_l0.1"
  run_one $S1 $F1 $T1 1    ema  0.9  0    0 "s1H8_o2_ema09_l1"
  run_one $S1 $F1 $T1 10   ema  0.9  0    0 "s1H8_o2_ema09_l10"
  # Option 3: threshold at 45° from init, sweep λ
  run_one $S1 $F1 $T1 0.1  init 0.95 45   0 "s1H8_o3_th45_l0.1"
  run_one $S1 $F1 $T1 1    init 0.95 45   0 "s1H8_o3_th45_l1"
  run_one $S1 $F1 $T1 10   init 0.95 45   0 "s1H8_o3_th45_l10"
) &
PID0=$!

(
  # Same sweeps on seq_2 HO_8
  run_one $S2 $F2 $T2 0.1  ema  0.9  0    1 "s2H8_o2_ema09_l0.1"
  run_one $S2 $F2 $T2 1    ema  0.9  0    1 "s2H8_o2_ema09_l1"
  run_one $S2 $F2 $T2 10   ema  0.9  0    1 "s2H8_o2_ema09_l10"
  run_one $S2 $F2 $T2 0.1  init 0.95 45   1 "s2H8_o3_th45_l0.1"
  run_one $S2 $F2 $T2 1    init 0.95 45   1 "s2H8_o3_th45_l1"
  run_one $S2 $F2 $T2 10   init 0.95 45   1 "s2H8_o3_th45_l10"
) &
PID1=$!

wait $PID0; wait $PID1
echo "all done"
