#!/usr/bin/env bash
# Q2 — difficulty of chosen test-frame sweep.
# For each of 6 scenes, shift the test frame within the fixed 9-frame
# window {F-4 .. F+4} and rerun v5/M1. Middle-frame (F itself) already
# exists except for seq_1_frame_277; we include it there.
#
# Scenes: seq_0_frame_135, seq_1_frame_185, seq_1_frame_277 (new),
#          seq_1_frame_438, seq_2_frame_160, seq_2_frame_300.
#
# Positions in window (index 0 = F-4, index 4 = F = middle):
#   idx 2 = F-2,  idx 3 = F-1,  idx 5 = F+1,  idx 6 = F+2.
set -u
cd /home/adnan/Desktop/mm3DGS
PY=/home/adnan/.conda/envs/mmir/bin/python
export PYTHONUNBUFFERED=1

LOG=/home/adnan/Desktop/mm3DGS/logs_v6_testframe_sweep
mkdir -p "$LOG"

run_one() {
    local scene="$1" F_center="$2" F_test="$3" gpu="$4"
    # Build train-frames list = {F-4..F+4} \ {F_test}
    local train_list=""
    for k in -4 -3 -2 -1 0 1 2 3 4; do
        local f=$((F_center + k))
        if [ $f -ne $F_test ]; then
            if [ -z "$train_list" ]; then
                train_list="$f"
            else
                train_list="$train_list,$f"
            fi
        fi
    done
    local tag="${scene}__F${F_test}"
    echo "[gpu $gpu] $tag starting (train=$train_list, test=$F_test)"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v6.train_frame_nvs \
        --scene "$scene" --test_frame "$F_test" --train_frames "$train_list" \
        --train_loops 0 \
        --iters 500 --loss_type mse_raw --target_n 20000 \
        --v6_milestone M1 \
        > "$LOG/${tag}.log" 2>&1
    echo "[gpu $gpu] $tag done"
}

# Per scene: 4 new positions (idx 2, 3, 5, 6). seq_1_frame_277 also needs
# the middle-index run since it wasn't in the prior 7-scene sweep.

GPU0_SCENES=( "seq_0_frame_135:135" "seq_1_frame_185:185" "seq_1_frame_277:277" )
GPU1_SCENES=( "seq_1_frame_438:438" "seq_2_frame_160:160" "seq_2_frame_300:300" )

do_scene() {
    local scene="$1" F_center="$2" gpu="$3" need_middle="$4"
    for off in -2 -1 1 2; do
        local F_test=$((F_center + off))
        run_one "$scene" "$F_center" "$F_test" "$gpu"
    done
    if [ "$need_middle" = "yes" ]; then
        run_one "$scene" "$F_center" "$F_center" "$gpu"
    fi
}

(
    for entry in "${GPU0_SCENES[@]}"; do
        scene="${entry%%:*}"; F="${entry##*:}"
        if [ "$scene" = "seq_1_frame_277" ]; then
            do_scene "$scene" "$F" 0 yes
        else
            do_scene "$scene" "$F" 0 no
        fi
    done
) &
PID0=$!

(
    for entry in "${GPU1_SCENES[@]}"; do
        scene="${entry%%:*}"; F="${entry##*:}"
        do_scene "$scene" "$F" 1 no
    done
) &
PID1=$!

wait $PID0; wait $PID1
echo "all done"
