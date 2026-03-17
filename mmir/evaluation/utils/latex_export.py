"""LaTeX table generation utilities."""

from typing import Dict, List, Optional, Tuple


def format_metric(val: Optional[float], fmt: str = ".3f") -> str:
    """Format a single metric value for LaTeX."""
    if val is None:
        return "---"
    return f"{val:{fmt}}"


def bold_best(values: List[Optional[float]], fmt: str = ".3f", higher_better: bool = True) -> List[str]:
    """Format a list of values, bolding the best one."""
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if not valid:
        return [format_metric(v, fmt) for v in values]

    best_idx = max(valid, key=lambda x: x[1] if higher_better else -x[1])[0]
    result = []
    for i, v in enumerate(values):
        s = format_metric(v, fmt)
        if i == best_idx and v is not None:
            s = r"\textbf{" + s + "}"
        result.append(s)
    return result


def bold_best_agg(
    agg: dict, key_a: str, key_b: str, fmt: str = ".3f", higher_better: bool = True,
) -> Tuple[str, str]:
    """Format two aggregate mean±std values, bolding the better one (by mean)."""
    def _fmt(key):
        if key in agg:
            return f"{agg[key]['mean']:{fmt}}$\\pm${agg[key]['std']:{fmt}}"
        return "---"

    a_mean = agg.get(key_a, {}).get("mean")
    b_mean = agg.get(key_b, {}).get("mean")

    a_str = _fmt(key_a)
    b_str = _fmt(key_b)

    if a_mean is not None and b_mean is not None:
        if higher_better:
            if a_mean >= b_mean:
                a_str = r"\textbf{" + a_str + "}"
            else:
                b_str = r"\textbf{" + b_str + "}"
        else:
            if a_mean <= b_mean:
                a_str = r"\textbf{" + a_str + "}"
            else:
                b_str = r"\textbf{" + b_str + "}"

    return a_str, b_str


def generate_training_ra_table(
    scene_metrics: Dict[str, dict],
    aggregate: dict,
    output_path: str,
):
    """Generate Table 1: Training RA metrics (ours vs benchmark).

    Args:
        scene_metrics: {scene_name: {ours_corr, ours_psnr, ours_ssim, bench_corr, bench_psnr, bench_ssim}}
        aggregate: {ours_corr: {mean, std}, bench_corr: {mean, std}, ...}
        output_path: path to save .tex file
    """
    lines = [
        r"\begin{tabular}{l cc cc cc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Correlation $\uparrow$} & \multicolumn{2}{c}{PSNR $\uparrow$} & \multicolumn{2}{c}{SSIM $\uparrow$} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"Scene & Ours & Sionna & Ours & Sionna & Ours & Sionna \\",
        r"\midrule",
    ]

    for scene_name, m in sorted(scene_metrics.items()):
        ours_corr, bench_corr = bold_best(
            [m.get("ours_corr"), m.get("bench_corr")], ".3f", higher_better=True
        )
        ours_psnr, bench_psnr = bold_best(
            [m.get("ours_psnr"), m.get("bench_psnr")], ".1f", higher_better=True
        )
        ours_ssim, bench_ssim = bold_best(
            [m.get("ours_ssim"), m.get("bench_ssim")], ".3f", higher_better=True
        )
        # Short scene name for table
        short = scene_name.replace("seq_", "S").replace("_frame_", "F")
        lines.append(
            f"  {short} & {ours_corr} & {bench_corr} & {ours_psnr} & {bench_psnr} & {ours_ssim} & {bench_ssim} \\\\"
        )

    lines.append(r"\midrule")
    # Aggregate row
    def _agg(key):
        if key in aggregate:
            return f"{aggregate[key]['mean']:.3f}$\\pm${aggregate[key]['std']:.3f}"
        return "---"

    lines.append(
        f"  Mean & {_agg('ours_corr')} & {_agg('bench_corr')} & "
        f"{_agg('ours_psnr')} & {_agg('bench_psnr')} & "
        f"{_agg('ours_ssim')} & {_agg('bench_ssim')} \\\\"
    )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    return output_path


def generate_training_ra_table_v2(
    scene_metrics: Dict[str, dict],
    aggregate: dict,
    output_path: str,
):
    """Generate Table 1 v2: Training RA metrics.

    Columns: RA Correlation, RA PSNR, RA SSIM, RA RMSE, ADC Log-Mag MSE, ADC MM-Mag MSE
    Single table with per-scene rows and a bolded Mean±Std row at the bottom.
    """
    _header = [
        r"& \multicolumn{2}{c}{RA Correlation $\uparrow$}"
        r" & \multicolumn{2}{c}{RA PSNR $\uparrow$}"
        r" & \multicolumn{2}{c}{RA SSIM $\uparrow$}"
        r" & \multicolumn{2}{c}{RA RMSE $\downarrow$}"
        r" & \multicolumn{2}{c}{ADC Log-Mag MSE $\downarrow$}"
        r" & \multicolumn{2}{c}{ADC MM-Mag MSE $\downarrow$} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}"
        r" \cmidrule(lr){8-9} \cmidrule(lr){10-11} \cmidrule(lr){12-13}",
        r"Scene & Ours & Sionna & Ours & Sionna & Ours & Sionna"
        r" & Ours & Sionna & Ours & Sionna & Ours & Sionna \\",
    ]

    lines = [
        r"% Table 1: Training RA — Per-scene results with aggregate",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Range--azimuth (RA) and ADC training evaluation across seven outdoor scenes. mmIR is compared against Sionna-RT~\cite{sionna}. Metrics: Pearson correlation (Corr), PSNR, SSIM, RMSE, and ADC-level magnitude error. Best in \textbf{bold}.}",
        r"\label{tab:training_ra}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{l cc cc cc cc cc cc}",
        r"\toprule",
        *_header,
        r"\midrule",
    ]

    for scene_name, m in sorted(scene_metrics.items()):
        ours_corr, bench_corr = bold_best(
            [m.get("ours_corr"), m.get("bench_corr")], ".3f", higher_better=True
        )
        ours_psnr, bench_psnr = bold_best(
            [m.get("ours_psnr"), m.get("bench_psnr")], ".1f", higher_better=True
        )
        ours_ssim, bench_ssim = bold_best(
            [m.get("ours_ssim"), m.get("bench_ssim")], ".3f", higher_better=True
        )
        ours_rmse, bench_rmse = bold_best(
            [m.get("ours_rmse"), m.get("bench_rmse")], ".4f", higher_better=False
        )
        ours_adc_log, bench_adc_log = bold_best(
            [m.get("ours_adc_log_mag_mse"), m.get("bench_adc_log_mag_mse")], ".4f", higher_better=False
        )
        ours_adc_mm, bench_adc_mm = bold_best(
            [m.get("ours_adc_minmax_mag_mse"), m.get("bench_adc_minmax_mag_mse")], ".4f", higher_better=False
        )
        short = scene_name.replace("seq_", "S").replace("_frame_", "F")
        lines.append(
            f"  {short} & {ours_corr} & {bench_corr}"
            f" & {ours_psnr} & {bench_psnr}"
            f" & {ours_ssim} & {bench_ssim}"
            f" & {ours_rmse} & {bench_rmse}"
            f" & {ours_adc_log} & {bench_adc_log}"
            f" & {ours_adc_mm} & {bench_adc_mm} \\\\"
        )

    # ---- Bolded Mean±Std row ----
    lines.append(r"\midrule")

    a_corr, b_corr = bold_best_agg(aggregate, "ours_corr", "bench_corr", ".3f", higher_better=True)
    a_psnr, b_psnr = bold_best_agg(aggregate, "ours_psnr", "bench_psnr", ".1f", higher_better=True)
    a_ssim, b_ssim = bold_best_agg(aggregate, "ours_ssim", "bench_ssim", ".3f", higher_better=True)
    a_rmse, b_rmse = bold_best_agg(aggregate, "ours_rmse", "bench_rmse", ".4f", higher_better=False)
    a_alog, b_alog = bold_best_agg(aggregate, "ours_adc_log_mag_mse", "bench_adc_log_mag_mse", ".4f", higher_better=False)
    a_amm, b_amm = bold_best_agg(aggregate, "ours_adc_minmax_mag_mse", "bench_adc_minmax_mag_mse", ".4f", higher_better=False)

    lines.append(
        f"  Mean & {a_corr} & {b_corr}"
        f" & {a_psnr} & {b_psnr}"
        f" & {a_ssim} & {b_ssim}"
        f" & {a_rmse} & {b_rmse}"
        f" & {a_alog} & {b_alog}"
        f" & {a_amm} & {b_amm} \\\\"
    )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    return output_path


def generate_radar_transfer_table(
    scene_metrics: Dict[str, dict],
    aggregate: dict,
    output_path: str,
):
    """Generate Table 2: Radar transfer metrics (Ours vs Benchmark).

    Columns: RA Correlation, RA PSNR, RA SSIM, RA RMSE, ADC Log-Mag MSE, ADC MM-Mag MSE
    Single double-column table with per-scene rows and a bolded Mean±Std row.
    """
    _header = [
        r"& \multicolumn{2}{c}{RA Correlation $\uparrow$}"
        r" & \multicolumn{2}{c}{RA PSNR $\uparrow$}"
        r" & \multicolumn{2}{c}{RA SSIM $\uparrow$}"
        r" & \multicolumn{2}{c}{RA RMSE $\downarrow$}"
        r" & \multicolumn{2}{c}{ADC Log-Mag MSE $\downarrow$}"
        r" & \multicolumn{2}{c}{ADC MM-Mag MSE $\downarrow$} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}"
        r" \cmidrule(lr){8-9} \cmidrule(lr){10-11} \cmidrule(lr){12-13}",
        r"Scene & Ours & Bench. & Ours & Bench. & Ours & Bench."
        r" & Ours & Bench. & Ours & Bench. & Ours & Bench. \\",
    ]

    lines = [
        r"% Table 2: Radar Transfer — Per-scene results with aggregate",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Cross-sensor transfer evaluation. Scene parameters optimized on the cascaded radar (12TX$\times$16RX, 86 unique azimuth virtual elements) are frozen and used to render ADC for the co-located single-chip radar (3TX$\times$4RX, 12 virtual elements) without re-training. Best in \textbf{bold}.}",
        r"\label{tab:radar_transfer}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{l cc cc cc cc cc cc}",
        r"\toprule",
        *_header,
        r"\midrule",
    ]

    for scene_name, m in sorted(scene_metrics.items()):
        short = scene_name.replace("seq_", "S").replace("_frame_", "F")

        ours_corr, bench_corr = bold_best(
            [m.get("transfer_cart_corr"), m.get("benchmark_cart_corr")],
            ".3f", higher_better=True,
        )
        ours_psnr, bench_psnr = bold_best(
            [m.get("transfer_cart_psnr"), m.get("benchmark_cart_psnr")],
            ".1f", higher_better=True,
        )
        ours_ssim, bench_ssim = bold_best(
            [m.get("transfer_cart_ssim"), m.get("benchmark_cart_ssim")],
            ".3f", higher_better=True,
        )
        ours_rmse, bench_rmse = bold_best(
            [m.get("transfer_cart_rmse"), m.get("benchmark_cart_rmse")],
            ".4f", higher_better=False,
        )
        ours_adc_log, bench_adc_log = bold_best(
            [m.get("transfer_adc_log_mag_mse"), m.get("benchmark_adc_log_mag_mse")],
            ".4f", higher_better=False,
        )
        ours_adc_mm, bench_adc_mm = bold_best(
            [m.get("transfer_adc_minmax_mag_mse"), m.get("benchmark_adc_minmax_mag_mse")],
            ".4f", higher_better=False,
        )

        lines.append(
            f"  {short}"
            f" & {ours_corr} & {bench_corr}"
            f" & {ours_psnr} & {bench_psnr}"
            f" & {ours_ssim} & {bench_ssim}"
            f" & {ours_rmse} & {bench_rmse}"
            f" & {ours_adc_log} & {bench_adc_log}"
            f" & {ours_adc_mm} & {bench_adc_mm} \\\\"
        )

    # ---- Bolded Mean±Std row ----
    lines.append(r"\midrule")

    a_corr, b_corr = bold_best_agg(aggregate, "transfer_cart_corr", "benchmark_cart_corr", ".3f", higher_better=True)
    a_psnr, b_psnr = bold_best_agg(aggregate, "transfer_cart_psnr", "benchmark_cart_psnr", ".1f", higher_better=True)
    a_ssim, b_ssim = bold_best_agg(aggregate, "transfer_cart_ssim", "benchmark_cart_ssim", ".3f", higher_better=True)
    a_rmse, b_rmse = bold_best_agg(aggregate, "transfer_cart_rmse", "benchmark_cart_rmse", ".4f", higher_better=False)
    a_alog, b_alog = bold_best_agg(aggregate, "transfer_adc_log_mag_mse", "benchmark_adc_log_mag_mse", ".4f", higher_better=False)
    a_amm, b_amm = bold_best_agg(aggregate, "transfer_adc_minmax_mag_mse", "benchmark_adc_minmax_mag_mse", ".4f", higher_better=False)

    lines.append(
        f"  Mean"
        f" & {a_corr} & {b_corr}"
        f" & {a_psnr} & {b_psnr}"
        f" & {a_ssim} & {b_ssim}"
        f" & {a_rmse} & {b_rmse}"
        f" & {a_alog} & {b_alog}"
        f" & {a_amm} & {b_amm} \\\\"
    )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    return output_path


def generate_3d_occupancy_table(
    scene_metrics: Dict[str, dict],
    aggregate: dict,
    output_path: str,
):
    """Generate Table 3: 3D occupancy metrics (Ours vs ColoRadar CASCADE)."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{3D occupancy evaluation against LiDAR ground truth. Baseline: ColoRadar-provided 3D point cloud from the cascaded radar's near-1D virtual aperture, where elevation is synthesized by extruding range--azimuth detections along the vertical axis due to minimal elevation resolution. Ours: dense virtual array (100TX$\times$100RX) with full 2D aperture enabling range--azimuth--elevation (RAE) imaging. Occupancy metrics use KD-tree nearest neighbors ($\tau{=}0.5$\,m). Best in \textbf{bold}.}",
        r"\label{tab:3d_occupancy}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{l cc cc cc cc cc cc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Precision $\uparrow$}"
        r" & \multicolumn{2}{c}{Recall $\uparrow$}"
        r" & \multicolumn{2}{c}{F1 $\uparrow$}"
        r" & \multicolumn{2}{c}{Accuracy $\uparrow$}"
        r" & \multicolumn{2}{c}{RMSE $\downarrow$}"
        r" & \multicolumn{2}{c}{R-CD $\downarrow$} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}"
        r" \cmidrule(lr){8-9} \cmidrule(lr){10-11} \cmidrule(lr){12-13}",
        r"Scene & Ours & ColoR & Ours & ColoR & Ours & ColoR"
        r" & Ours & ColoR & Ours & ColoR & Ours & ColoR \\",
        r"\midrule",
    ]

    # Collect per-scene F1 values for aggregate computation
    ours_f1_vals = []
    color_f1_vals = []

    for scene_name, m in sorted(scene_metrics.items()):
        short = scene_name.replace("seq_", "S").replace("_frame_", "F")

        # Ours
        o_prec = m.get("distance_precision")
        o_rec = m.get("distance_recall")
        o_acc = m.get("distance_accuracy")
        o_rmse = m.get("pc_rmse")
        o_rcd = m.get("relative_chamfer_distance")
        o_p = o_prec if o_prec is not None else 0.0
        o_r = o_rec if o_rec is not None else 0.0
        o_f1 = 2 * o_p * o_r / (o_p + o_r) if (o_p + o_r) > 0 else None

        # ColoRadar
        c_prec = m.get("coloradar_distance_precision")
        c_rec = m.get("coloradar_distance_recall")
        c_acc = m.get("coloradar_distance_accuracy")
        c_rmse = m.get("coloradar_pc_rmse")
        c_rcd = m.get("coloradar_relative_chamfer_distance")
        c_p = c_prec if c_prec is not None else 0.0
        c_r = c_rec if c_rec is not None else 0.0
        c_f1 = 2 * c_p * c_r / (c_p + c_r) if (c_p + c_r) > 0 else None

        if o_f1 is not None:
            ours_f1_vals.append(o_f1)
        if c_f1 is not None:
            color_f1_vals.append(c_f1)

        prec_o, prec_c = bold_best([o_prec, c_prec], ".3f", higher_better=True)
        rec_o, rec_c = bold_best([o_rec, c_rec], ".3f", higher_better=True)
        f1_o, f1_c = bold_best([o_f1, c_f1], ".3f", higher_better=True)
        acc_o, acc_c = bold_best([o_acc, c_acc], ".3f", higher_better=True)
        rmse_o, rmse_c = bold_best([o_rmse, c_rmse], ".3f", higher_better=False)
        rcd_o, rcd_c = bold_best([o_rcd, c_rcd], ".3f", higher_better=False)

        lines.append(
            f"  {short} & {prec_o} & {prec_c}"
            f" & {rec_o} & {rec_c}"
            f" & {f1_o} & {f1_c}"
            f" & {acc_o} & {acc_c}"
            f" & {rmse_o} & {rmse_c}"
            f" & {rcd_o} & {rcd_c} \\\\"
        )

    # ---- Bolded Mean±Std row ----
    lines.append(r"\midrule")

    a_prec, b_prec = bold_best_agg(aggregate, "distance_precision", "coloradar_distance_precision", ".3f", higher_better=True)
    a_rec, b_rec = bold_best_agg(aggregate, "distance_recall", "coloradar_distance_recall", ".3f", higher_better=True)
    a_acc, b_acc = bold_best_agg(aggregate, "distance_accuracy", "coloradar_distance_accuracy", ".3f", higher_better=True)
    a_rmse, b_rmse = bold_best_agg(aggregate, "pc_rmse", "coloradar_pc_rmse", ".3f", higher_better=False)
    a_rcd, b_rcd = bold_best_agg(aggregate, "relative_chamfer_distance", "coloradar_relative_chamfer_distance", ".3f", higher_better=False)

    # Compute F1 aggregate with bolding
    import numpy as _np
    if ours_f1_vals:
        ours_f1_mean = float(_np.mean(ours_f1_vals))
        ours_f1_str = f"{ours_f1_mean:.3f}$\\pm${float(_np.std(ours_f1_vals)):.3f}"
    else:
        ours_f1_mean = None
        ours_f1_str = "---"
    if color_f1_vals:
        color_f1_mean = float(_np.mean(color_f1_vals))
        color_f1_str = f"{color_f1_mean:.3f}$\\pm${float(_np.std(color_f1_vals)):.3f}"
    else:
        color_f1_mean = None
        color_f1_str = "---"
    if ours_f1_mean is not None and color_f1_mean is not None:
        if ours_f1_mean >= color_f1_mean:
            ours_f1_str = r"\textbf{" + ours_f1_str + "}"
        else:
            color_f1_str = r"\textbf{" + color_f1_str + "}"

    lines.append(
        f"  Mean & {a_prec} & {b_prec}"
        f" & {a_rec} & {b_rec}"
        f" & {ours_f1_str} & {color_f1_str}"
        f" & {a_acc} & {b_acc}"
        f" & {a_rmse} & {b_rmse}"
        f" & {a_rcd} & {b_rcd} \\\\"
    )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    return output_path


def generate_3d_reconstruction_table(
    scene_metrics: Dict[str, dict],
    aggregate: dict,
    output_path: str,
):
    """Generate Table 4: 3D dense reconstruction metrics."""
    lines = [
        r"\begin{tabular}{l cccc cc c}",
        r"\toprule",
        r"Scene & Precision & Recall & F1 & Accuracy & RMSE & R-CD & \#Views \\",
        r"\midrule",
    ]

    for scene_name, m in sorted(scene_metrics.items()):
        short = scene_name.replace("seq_", "S").replace("_frame_", "F")
        prec_val = m.get("distance_precision", m.get("precision"))
        rec_val = m.get("distance_recall", m.get("recall"))
        acc_val = m.get("distance_accuracy", m.get("accuracy"))
        rmse_val = m.get("pc_rmse", m.get("rmse"))
        rcd_val = m.get("relative_chamfer_distance", m.get("relative_chamfer"))
        # Compute F1 from precision/recall
        p = prec_val if prec_val is not None else 0.0
        r = rec_val if rec_val is not None else 0.0
        f1_val = 2 * p * r / (p + r) if (p + r) > 0 else None
        prec = format_metric(prec_val, ".3f")
        rec = format_metric(rec_val, ".3f")
        f1 = format_metric(f1_val, ".3f")
        acc = format_metric(acc_val, ".3f")
        rmse = format_metric(rmse_val, ".3f")
        rcd = format_metric(rcd_val, ".3f")
        n_frames = format_metric(m.get("num_frames"), ".0f")
        lines.append(f"  {short} & {prec} & {rec} & {f1} & {acc} & {rmse} & {rcd} & {n_frames} \\\\")

    lines.append(r"\midrule")

    def _agg(key, fmt=".3f"):
        if key in aggregate:
            return f"{aggregate[key]['mean']:{fmt}}$\\pm${aggregate[key]['std']:{fmt}}"
        return "---"

    lines.append(
        f"  Mean & {_agg('distance_precision')} & {_agg('distance_recall')} & --- & "
        f"{_agg('distance_accuracy')} & {_agg('pc_rmse')} & {_agg('relative_chamfer_distance')} & --- \\\\"
    )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    return output_path
