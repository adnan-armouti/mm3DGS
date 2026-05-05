"""Diagnostic: borrow GT RA phase into pred RA and re-measure all metrics.

Construction:
    RA_pred (complex) and RA_gt (complex) are both computed via the trainer's
    azimuth FFT pipeline (Hann_az + FFT_az).
    Hybrid RA: RA_h = |RA_pred| · exp(j · arg(RA_gt))
        — magnitude from pred (the trained-for quantity), phase from GT.

    IFFT_az(RA_h) → CRP_h (in Hann_az × CRP domain, same as our eval).
    IFFT_range(CRP_h) → ADC_h.

Then we compute the SAME metric set we use in the main eval against the GT,
side-by-side with the unmodified pred numbers. This is the upper bound on
how well a magnitude-only-trained renderer could ever look on complex
metrics if we were allowed to borrow GT phase wholesale (we aren't, for the
paper — this is a thought experiment).

Also runs the inverse experiment: |RA_gt| · exp(j · arg(RA_pred)) — what if
we had perfect magnitude but kept the renderer's phase? This isolates how
much phase content our renderer actually has.
"""
import sys, os, json
sys.path.insert(0, '/home/adnan/Desktop/mm3DGS')
import numpy as np
from mmir.data.ra_utils import txrx_to_virtual_array_numpy
from mmir.evaluation.eval_crp_adc import (
    PAPER_SCENES_FULL, _scene_gt_adc_path, _scene_rendered_rp_test,
    _scene_rendered_rp_train,
    N_VX, K_RANGE, NEAR_FIELD_BINS,
)

START_BIN = NEAR_FIELD_BINS    # 15


def hybrid_experiment(adc_gt_txrx, rp_pred_txrx, start_bin=START_BIN):
    """Compute pred / hybrid_phase / hybrid_mag CRP+ADC arrays."""
    # ── GT CRP (trainer loss-domain: Hann_range × FFT_range × ADC) ──
    vx_gt = txrx_to_virtual_array_numpy(adc_gt_txrx)
    crp_gt = np.fft.fft(vx_gt * np.hanning(K_RANGE)[None, :], axis=-1)

    # ── Pred CRP (renderer's emitted CRP, already Hann_range domain) ──
    crp_pred = txrx_to_virtual_array_numpy(rp_pred_txrx)

    # ── Near-field mask both ──
    crp_gt = crp_gt.copy();   crp_gt[:, :start_bin] = 0
    crp_pred = crp_pred.copy(); crp_pred[:, :start_bin] = 0

    # ── Apply Hann_az to put both in trainer's loss domain ──
    hann_a = np.hanning(N_VX)
    crp_gt_h = crp_gt * hann_a[:, None]
    crp_pred_h = crp_pred * hann_a[:, None]

    # ── FFT_az on both (n=86 for clean invertibility, no shifts) ──
    ra_gt = np.fft.fft(crp_gt_h, axis=0)
    ra_pred = np.fft.fft(crp_pred_h, axis=0)

    # ── Hybrid #1: |RA_pred| · exp(j · arg(RA_gt))  ── pred mag, GT phase
    ra_hybrid_phase = np.abs(ra_pred) * np.exp(1j * np.angle(ra_gt))
    crp_hybrid_phase = np.fft.ifft(ra_hybrid_phase, axis=0)

    # ── Hybrid #2: |RA_gt| · exp(j · arg(RA_pred))  ── GT mag, pred phase
    ra_hybrid_mag = np.abs(ra_gt) * np.exp(1j * np.angle(ra_pred))
    crp_hybrid_mag = np.fft.ifft(ra_hybrid_mag, axis=0)

    # ── ADCs from each CRP ──
    adc_gt = np.fft.ifft(crp_gt_h, axis=-1)
    adc_pred = np.fft.ifft(crp_pred_h, axis=-1)
    adc_hybrid_phase = np.fft.ifft(crp_hybrid_phase, axis=-1)
    adc_hybrid_mag = np.fft.ifft(crp_hybrid_mag, axis=-1)

    return dict(
        crp_gt=crp_gt_h, crp_pred=crp_pred_h,
        crp_hyb_phase=crp_hybrid_phase, crp_hyb_mag=crp_hybrid_mag,
        adc_gt=adc_gt, adc_pred=adc_pred,
        adc_hyb_phase=adc_hybrid_phase, adc_hyb_mag=adc_hybrid_mag,
    )


def metrics(pred, gt, start_bin=START_BIN, crop_range=True):
    """Same metric set as the main eval."""
    if crop_range:
        p = pred[:, start_bin:]; g = gt[:, start_bin:]
    else:
        p = pred; g = gt
    pf = p.ravel(); gf = g.ravel()

    # |·| Pearson with min-max norm
    def _mm(x):
        x = np.abs(x).astype(np.float64)
        mn, mx = x.min(), x.max()
        return (x - mn) / max(mx - mn, 1e-30)
    mag_corr = float(np.corrcoef(_mm(p).ravel(), _mm(g).ravel())[0, 1])

    # complex |ρ|
    inner = np.vdot(gf, pf)
    rho = float(np.abs(inner) / (np.linalg.norm(gf) * np.linalg.norm(pf) + 1e-30))

    # gain-matched RecSNR + phase RMSE
    pp = float(np.vdot(pf, pf).real)
    if pp > 1e-30:
        c = np.conj(inner) / pp
        p_a = p * c
    else:
        p_a = p
    diff = p_a - g
    nmse = float(np.abs(diff).ravel() @ np.abs(diff).ravel() /
                  max(np.abs(g).ravel() @ np.abs(g).ravel(), 1e-30))
    rec_snr = -10 * np.log10(nmse) if nmse > 0 else float('inf')

    abs_g = np.abs(g)
    phi_diff = np.angle(p_a * np.conj(g))
    w = (abs_g ** 2).astype(np.float64)
    if w.sum() > 1e-30:
        phase_rmse_rad = float(np.sqrt((w * phi_diff ** 2).sum() / w.sum()))
    else:
        phase_rmse_rad = float('nan')
    return {
        'mag_corr': mag_corr,
        'rho_C': rho,
        'rec_snr_db': rec_snr,
        'phase_rmse_deg': float(np.degrees(phase_rmse_rad)),
    }


def fmt(m):
    return (f"|.|={m['mag_corr']:.3f}  |ρ|={m['rho_C']:.3f}  "
            f"RecSNR={m['rec_snr_db']:.2f}dB  σ_φ={m['phase_rmse_deg']:.1f}°")


def _run_split(split_name: str, frames_per_scene_iter):
    """frames_per_scene_iter yields (scene, F, gt_path, rp_path)."""
    rows_crp = {'pred': [], 'hyb_phase': [], 'hyb_mag': []}
    rows_adc = {'pred': [], 'hyb_phase': [], 'hyb_mag': []}
    n_scenes_done = 0
    for scene, F, gt_p, rp_p in frames_per_scene_iter:
        if not (os.path.isfile(gt_p) and os.path.isfile(rp_p)):
            continue
        gt_raw = np.load(gt_p)
        adc_txrx = gt_raw[0].transpose(1, 0, 2).astype(np.complex128)
        rp_pred = np.load(rp_p).astype(np.complex128)

        out = hybrid_experiment(adc_txrx, rp_pred)

        rows_crp['pred'].append(metrics(out['crp_pred'], out['crp_gt']))
        rows_crp['hyb_phase'].append(metrics(out['crp_hyb_phase'], out['crp_gt']))
        rows_crp['hyb_mag'].append(metrics(out['crp_hyb_mag'], out['crp_gt']))
        rows_adc['pred'].append(metrics(out['adc_pred'], out['adc_gt'], crop_range=False))
        rows_adc['hyb_phase'].append(metrics(out['adc_hyb_phase'], out['adc_gt'], crop_range=False))
        rows_adc['hyb_mag'].append(metrics(out['adc_hyb_mag'], out['adc_gt'], crop_range=False))
        n_scenes_done += 1

    print(f"\n{'='*90}")
    print(f"AGGREGATE — {split_name}  (mean over {n_scenes_done} frames)")
    print(f"{'='*90}")
    for dom, rows in (("CRP", rows_crp), ("ADC", rows_adc)):
        for label, lst in rows.items():
            if not lst:
                continue
            mean = {k: float(np.mean([r[k] for r in lst])) for k in lst[0]}
            print(f"  {dom}  {label:<10s}: {fmt(mean)}")
        print()
    return rows_crp, rows_adc


def main():
    print("=" * 90)
    print("Phase-substitution experiment — TRAIN vs TEST")
    print("=" * 90)
    print("Variants:")
    print("  PRED        : raw renderer output (no phase correction)")
    print("  HYB_PHASE   : RA_h = |RA_pred| · exp(j·arg(RA_gt))  ← borrow GT phase")
    print("  HYB_MAG     : RA_h = |RA_gt| · exp(j·arg(RA_pred))  ← borrow GT magnitude")
    print()
    print("Theory under test: HYB_PHASE |ρ| ceiling is bounded by |RA| Pearson.")
    print("If true: train (RA Pearson ≈ 0.8) should give HIGHER HYB_PHASE |ρ|")
    print("         than test (RA Pearson ≈ 0.6, HYB_PHASE |ρ| was 0.58).")

    # ── TEST split ──
    def test_iter():
        for scene, F, _ in PAPER_SCENES_FULL:
            yield (scene, F,
                    _scene_gt_adc_path(scene, F),
                    _scene_rendered_rp_test(scene, F))

    # ── TRAIN split (8 train frames per scene = 48 frames total) ──
    def train_iter():
        for scene, F_test, train_frames in PAPER_SCENES_FULL:
            for F in train_frames:
                yield (scene, F,
                        _scene_gt_adc_path(scene, F),
                        _scene_rendered_rp_train(scene, F_test, F))

    test_crp, test_adc = _run_split("TEST  (held-out, 6 frames)", test_iter())
    train_crp, train_adc = _run_split("TRAIN (8 views/scene, 48 frames)", train_iter())

    # ── Side-by-side summary ──
    print("=" * 90)
    print("SIDE-BY-SIDE: ceiling test")
    print("=" * 90)
    def m(rows, name, key):
        if not rows[name]: return float('nan')
        return float(np.mean([r[key] for r in rows[name]]))
    print(f"\n{'':22s}{'TRAIN':^28s}{'TEST':^28s}")
    print(f"{'':22s}{'(|RA| corr ≈ 0.8)':^28s}{'(|RA| corr ≈ 0.6)':^28s}")
    print(f"{'-' * 78}")
    for name in ('pred', 'hyb_phase', 'hyb_mag'):
        for dom, rows_train, rows_test in (("CRP", train_crp, test_crp),
                                              ("ADC", train_adc, test_adc)):
            tr_rho = m(rows_train, name, 'rho_C')
            te_rho = m(rows_test, name, 'rho_C')
            tr_phi = m(rows_train, name, 'phase_rmse_deg')
            te_phi = m(rows_test, name, 'phase_rmse_deg')
            print(f"  {dom} {name:<10s}  |ρ|={tr_rho:.3f}  σ_φ={tr_phi:5.1f}°"
                  f"     |ρ|={te_rho:.3f}  σ_φ={te_phi:5.1f}°")
        print()


if __name__ == "__main__":
    main()
