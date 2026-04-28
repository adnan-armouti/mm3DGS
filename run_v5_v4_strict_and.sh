#!/usr/bin/env bash
# Phase B1/B2 — strict-AND visibility init (8 train poses for B1; +test
# pose for B2 as informational ablation). All other knobs at v5_v4
# default (combo_jitter recipe). 6 scenes per variant; GPU 0 = B1,
# GPU 1 = B2 so the two variants run in parallel.
set -u
cd /home/adnan/Desktop/mm3DGS
PY=/home/adnan/.conda/envs/mmir/bin/python
export PYTHONUNBUFFERED=1
LOG=/home/adnan/Desktop/mm3DGS/logs_v5_v4/strict_and_6scene
mkdir -p "$LOG"

SCENES=( "seq_0_frame_135:135" "seq_1_frame_185:185" "seq_1_frame_438:438"
         "seq_2_frame_105:105" "seq_2_frame_160:160" "seq_2_frame_300:300" )

run_one() {
    local scene="$1" F="$2" variant="$3" gpu="$4"
    local TRAIN="$((F-4)),$((F-3)),$((F-2)),$((F-1)),$((F+1)),$((F+2)),$((F+3)),$((F+4))"
    local tag="${scene}__${variant}"
    if [ -f "${LOG}/${tag}.log" ] && grep -q "\[final\]" "${LOG}/${tag}.log" 2>/dev/null; then
        echo "[gpu $gpu] $tag SKIP"; return
    fi
    echo "[gpu $gpu] $tag starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v5_v4.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$TRAIN" \
        --init_variant "$variant" \
        > "$LOG/${tag}.log" 2>&1
    echo "[gpu $gpu] $tag done (exit=$?)"
}

run_all() {
    local gpu="$1" variant="$2"
    for entry in "${SCENES[@]}"; do
        IFS=':' read -r scene F <<< "$entry"
        run_one "$scene" "$F" "$variant" "$gpu"
    done
}

( run_all 0 B1_strict_and_train ) & PID0=$!
( run_all 1 B2_strict_and_with_test ) & PID1=$!
wait $PID0; wait $PID1
echo "all done"
