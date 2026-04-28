#!/usr/bin/env bash
# MLP-A: pose-conditioned per-Gaussian deformation MLP. lr=5e-4, warmup=50,
# default L2 anchor on Δp + L1 sparsity on Δα. All other knobs at v5_v4
# combo_jitter recipe (M0 tightened FOV).
set -u
cd /home/adnan/Desktop/mm3DGS
PY=/home/adnan/.conda/envs/mmir/bin/python
export PYTHONUNBUFFERED=1
LOG=/home/adnan/Desktop/mm3DGS/logs_v5_v4/mlpa_6scene
mkdir -p "$LOG"
run_one() {
    local scene="$1" F="$2" gpu="$3"
    local TRAIN="$((F-4)),$((F-3)),$((F-2)),$((F-1)),$((F+1)),$((F+2)),$((F+3)),$((F+4))"
    if [ -f "${LOG}/${scene}.log" ] && grep -q "\[final\]" "${LOG}/${scene}.log" 2>/dev/null; then
        echo "[gpu $gpu] $scene SKIP"; return
    fi
    echo "[gpu $gpu] $scene starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v5_v4.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$TRAIN" \
        --mlp_a_lr 5e-4 --mlp_a_warmup_iters 50 \
        > "$LOG/${scene}.log" 2>&1
    echo "[gpu $gpu] $scene done (exit=$?)"
}
GPU0=( "seq_0_frame_135:135" "seq_1_frame_185:185" "seq_1_frame_438:438" )
GPU1=( "seq_2_frame_105:105" "seq_2_frame_160:160" "seq_2_frame_300:300" )
run_gpu() { local gpu="$1"; shift; for entry in "$@"; do IFS=':' read -r s F <<< "$entry"; run_one "$s" "$F" "$gpu"; done; }
( run_gpu 0 "${GPU0[@]}" ) & PID0=$!
( run_gpu 1 "${GPU1[@]}" ) & PID1=$!
wait $PID0; wait $PID1
echo "all done"
