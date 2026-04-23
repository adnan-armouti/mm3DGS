#!/usr/bin/env bash
# Stage 3 of data_v2 plan (see md/extended_window_data_v2_plan.md §4).
#
# For each of 6 scenes, run two variants:
#   - F±4 (HO_8 bracket) training on data_v2 — feeds gate G3.1.
#   - Extended asymmetric-window training on data_v2 — the experiment.
#
# Loss is v5 mse_raw (milestone M1), first chirp only, 500 iters,
# target_n=20000. PNG saving enabled so we have visual artefacts for
# inspection alongside the numbers.
set -u
cd /home/adnan/Desktop/mm3DGS
PY=/home/adnan/.conda/envs/mmir/bin/python
export PYTHONUNBUFFERED=1

LOG=/home/adnan/Desktop/mm3DGS/logs_data_v2_train
mkdir -p "$LOG"

# Comma-join a numeric range-list: pass the list values as args.
join_commas() { local IFS=','; echo "$*"; }

range_exclude_center() {
    # $1=lo, $2=hi, $3=center -> inclusive integer range with center removed
    local lo="$1" hi="$2" c="$3"
    local out=""
    for ((k=lo; k<=hi; k++)); do
        if [ "$k" -eq "$c" ]; then continue; fi
        out="${out:+$out,}$k"
    done
    echo "$out"
}

run_one() {
    local scene="$1" F="$2" train="$3" tag="$4" gpu="$5"
    echo "[gpu $gpu] ${scene}__${tag} starting"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m mm25DGS_v6.train_frame_nvs \
        --scene "$scene" --test_frame "$F" --train_frames "$train" \
        --train_loops 0 \
        --iters 500 --loss_type mse_raw --target_n 20000 \
        --v6_milestone M1 \
        --data_root /home/adnan/Desktop/mm3DGS/data_v2 \
        --save_ra_pngs \
        > "$LOG/${scene}__${tag}.log" 2>&1
    echo "[gpu $gpu] ${scene}__${tag} done (exit=$?)"
}

# Per-scene definitions: "scene:F:ext_lo:ext_hi"
SCENES_GPU0=(
    "seq_0_frame_135:135:-21:18"
    "seq_1_frame_185:185:-25:25"
    "seq_1_frame_277:277:-25:30"
)
SCENES_GPU1=(
    "seq_1_frame_438:438:-35:30"
    "seq_2_frame_160:160:-15:15"
    "seq_2_frame_300:300:-19:24"
)

run_gpu_variant() {
    local gpu="$1"; shift
    for entry in "$@"; do
        IFS=':' read -r scene F lo hi <<< "$entry"
        # F±4 bracket
        local train_fpm4="$(range_exclude_center $((F-4)) $((F+4)) $F)"
        run_one "$scene" "$F" "$train_fpm4" "fpm4" "$gpu"
        # Extended asymmetric bracket
        local train_ext="$(range_exclude_center $((F+lo)) $((F+hi)) $F)"
        run_one "$scene" "$F" "$train_ext" "ext" "$gpu"
    done
}

( run_gpu_variant 0 "${SCENES_GPU0[@]}" ) &
PID0=$!
( run_gpu_variant 1 "${SCENES_GPU1[@]}" ) &
PID1=$!

wait $PID0; wait $PID1
echo "all done"
