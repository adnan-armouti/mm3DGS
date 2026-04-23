#!/usr/bin/env bash
# Two-variant sweep × 7 scenes, 500 iters, with RA PNG + cc_history.png
# saved per run:
#
#   v5_rerun    : v6_milestone=M1 (v5 mse_raw, no loss change)
#   hybrid_mag  : v6_milestone=M1_5, loss_variant=hybrid_mag, λ=1.0
#                 i.e. L = L_v5 + λ · Σ_{n,r} (|v_p[n,r]| − |v_g[n,r]|)²
#                 where the |·| is on 192-virt per-antenna per-range mags.
#
# Outputs land under mm25DGS_v6/output_frame_nvs/<scene>_..._v6M1 and
# <scene>_..._v6M1_5_hybrid_mag respectively. The v5_rerun dirs collide
# in name with the M1 diagnostic runs we already have — to keep the
# fresh RA PNGs + cc_history saved here distinguishable from the older
# M1 runs, the script renames the existing _v6M1 dirs to
# _v6M1_nopngs_archived before launching the rerun.
set -u
cd /home/adnan/Desktop/mm3DGS
PY=/home/adnan/.conda/envs/mmir/bin/python
export PYTHONUNBUFFERED=1

LOG=/home/adnan/Desktop/mm3DGS/logs_v6_hybrid_sweep
mkdir -p "$LOG"

# Archive any existing _v6M1 dirs (no PNGs) so the rerun's PNG-enabled
# output dirs don't collide.
for d in mm25DGS_v6/output_frame_nvs/seq_*_v6M1; do
  [ -d "$d" ] || continue
  # Skip dirs that already end in _v6M1_X where X != _5_* etc
  case "$d" in
    *_v6M1_5*) continue ;;  # skip M1.5 dirs
    *_v6M1)    mv "$d" "${d}_nopngs_archived" ;;
  esac
done

train_frames_for() {
    local F="$1"
    echo "$((F-4)),$((F-3)),$((F-2)),$((F-1)),$((F+1)),$((F+2)),$((F+3)),$((F+4))"
}

run_v5_rerun() {
    local scene="$1" F="$2" gpu="$3"
    local TRAIN=$(train_frames_for "$F")
    local tag="${scene}__v5_rerun"
    echo "[gpu $gpu] $tag starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v6.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$TRAIN" \
        --train_loops 0 \
        --iters 500 --loss_type mse_raw --target_n 20000 \
        --v6_milestone M1 \
        --save_ra_pngs \
        > "$LOG/${tag}.log" 2>&1
    echo "[gpu $gpu] $tag done"
}

run_hybrid() {
    local scene="$1" F="$2" gpu="$3"
    local TRAIN=$(train_frames_for "$F")
    local tag="${scene}__hybrid_mag"
    echo "[gpu $gpu] $tag starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v6.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$TRAIN" \
        --train_loops 0 \
        --iters 500 --loss_type mse_raw --target_n 20000 \
        --v6_milestone M1_5 --v6_loss_variant hybrid_mag --lambda_mag 1.0 \
        --save_ra_pngs \
        > "$LOG/${tag}.log" 2>&1
    echo "[gpu $gpu] $tag done"
}

SCENES_GPU0=(
  "seq_0_frame_135 135"
  "seq_0_frame_390 390"
  "seq_1_frame_185 185"
  "seq_1_frame_438 438"
)
SCENES_GPU1=(
  "seq_2_frame_105 105"
  "seq_2_frame_160 160"
  "seq_2_frame_300 300"
)

# GPU 0: all v5_rerun then all hybrid_mag for its 4 scenes (sequential
# within GPU). GPU 1: same for its 3 scenes.
(
  for sc in "${SCENES_GPU0[@]}"; do run_v5_rerun $sc 0; done
  for sc in "${SCENES_GPU0[@]}"; do run_hybrid $sc 0; done
) &
PID0=$!
(
  for sc in "${SCENES_GPU1[@]}"; do run_v5_rerun $sc 1; done
  for sc in "${SCENES_GPU1[@]}"; do run_hybrid $sc 1; done
) &
PID1=$!

wait $PID0; wait $PID1
echo "all done"
