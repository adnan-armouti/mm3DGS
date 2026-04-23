#!/usr/bin/env bash
# data_v2 pipeline — stages 1 (preproc) + 2 (alignment) for 5 scenes,
# parallelised across 2 GPUs. seq_2_frame_160 is already complete
# (smoke-tested), so it's excluded here.
#
# Per-scene params (from md/extended_window_data_v2_plan.md §4 Stage 1):
#   seq_0_frame_135: N_cas=43 (±21), N_lid=50
#   seq_1_frame_185: N_cas=51 (±25), N_lid=52
#   seq_1_frame_277: N_cas=61 (±30), N_lid=62
#   seq_1_frame_438: N_cas=71 (±35), N_lid=72
#   seq_2_frame_300: N_cas=49 (±24), N_lid=50
set -u
cd /home/adnan/Desktop/mm3DGS
PY=/home/adnan/.conda/envs/mmir/bin/python
export PYTHONUNBUFFERED=1

LOG=/home/adnan/Desktop/mm3DGS/logs_data_v2
mkdir -p "$LOG"

DATASET_ROOT=/home/adnan/Documents/Data/coloRadar/raw/kitti/2_28_2021_outdoors_run
CALIB=/home/adnan/Desktop/mm3DGS/mm25DGS_v6/preprocessing/calib

preproc_scene() {
    local seq="$1" F="$2" N_cas="$3" N_lid="$4" gpu="$5"
    local tag="seq_${seq}_frame_${F}__preproc"
    echo "[gpu $gpu] $tag starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v6.preprocessing.preproc all \
        --seq "$seq" --frame "$F" --num-radar-frames "$N_cas" \
        --dataset-dir "$DATASET_ROOT" \
        --calib-path "$CALIB" \
        --out-root data_v2 \
        --cascade --num-lidar-frames "$N_lid" \
        --buffer-distance 1.0 --normals-radius 0.1 --remove-behind-radar --verbose \
        > "$LOG/${tag}.log" 2>&1
    echo "[gpu $gpu] $tag done (exit=$?)"
}

align_scene() {
    local seq="$1" F="$2" gpu="$3"
    local scene="seq_${seq}_frame_${F}"
    local tag="${scene}__align"
    echo "[gpu $gpu] $tag starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v6.preprocessing.alignment.run_alignment \
        --scene "$scene" --data-root data_v2 \
        > "$LOG/${tag}.log" 2>&1
    echo "[gpu $gpu] $tag done (exit=$?)"
}

# Scenes per GPU (comma-separated: seq,F,N_cas,N_lid)
GPU0_SCENES=(
    "0,135,43,50"
    "1,185,51,52"
    "1,277,61,62"
)
GPU1_SCENES=(
    "1,438,71,72"
    "2,300,49,50"
)

run_gpu_pipeline() {
    local gpu="$1"; shift
    for entry in "$@"; do
        IFS=',' read -r seq F N_cas N_lid <<< "$entry"
        preproc_scene "$seq" "$F" "$N_cas" "$N_lid" "$gpu"
    done
    for entry in "$@"; do
        IFS=',' read -r seq F N_cas N_lid <<< "$entry"
        align_scene "$seq" "$F" "$gpu"
    done
}

( run_gpu_pipeline 0 "${GPU0_SCENES[@]}" ) &
PID0=$!
( run_gpu_pipeline 1 "${GPU1_SCENES[@]}" ) &
PID1=$!

wait $PID0; wait $PID1
echo "all done"
