#!/usr/bin/env bash
# Q3 — baseline-binned Gram loss sweep (3 variants × 7 scenes).
#
#   baseline_binned                 : uniform weights, keep b=0 bin
#   baseline_binned_wiener          : Wiener σ_θ=0.03 rad weights,
#                                      keep b=0 bin
#   baseline_binned_wiener_skipb0   : Wiener weights + skip b=0 bin
#                                      (expected: magnitude-less ⇒
#                                      worse, useful ablation)
set -u
cd /home/adnan/Desktop/mm3DGS
PY=/home/adnan/.conda/envs/mmir/bin/python
export PYTHONUNBUFFERED=1

LOG=/home/adnan/Desktop/mm3DGS/logs_v6_baseline_binned_sweep
mkdir -p "$LOG"

VARIANTS=(
  baseline_binned
  baseline_binned_wiener
  baseline_binned_wiener_skipb0
)

train_frames_for() {
    local F="$1"
    echo "$((F-4)),$((F-3)),$((F-2)),$((F-1)),$((F+1)),$((F+2)),$((F+3)),$((F+4))"
}

run_one() {
    local scene="$1" F="$2" variant="$3" gpu="$4"
    local TRAIN=$(train_frames_for "$F")
    local tag="${scene}__${variant}"
    echo "[gpu $gpu] $tag starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v6.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$TRAIN" \
        --train_loops 0 \
        --iters 500 --loss_type mse_raw --target_n 20000 \
        --v6_milestone M1_5 --v6_loss_variant "$variant" \
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

(
  for variant in "${VARIANTS[@]}"; do
    for sc in "${SCENES_GPU0[@]}"; do
      run_one $sc "$variant" 0
    done
  done
) &
PID0=$!

(
  for variant in "${VARIANTS[@]}"; do
    for sc in "${SCENES_GPU1[@]}"; do
      run_one $sc "$variant" 1
    done
  done
) &
PID1=$!

wait $PID0; wait $PID1
echo "all done"
