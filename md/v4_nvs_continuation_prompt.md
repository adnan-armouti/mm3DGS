# mm3DGS v4 NVS — continuation prompt for next Claude Code session

Paste the following into a new Claude Code session to resume the work:

---

```
I'm continuing work on mm3DGS v4 NVS (novel view synthesis with a differentiable
mmWave radar renderer). Please start by reading the status report at
/home/adnan/Desktop/mm3DGS/md/v4_nvs_status_report.md which summarizes everything
we've established so far. The relevant companion docs in /md/ are:

- v4_nvs_plan.md            — original NVS design
- v4_nvs_diagnosis.md       — early failure-mode diagnostics
- v4_nvs_param_analysis.md  — parameter-space comparisons
- v4_nvs_scenario_diff.md   — output-space diagnostics + reg experiments
- v4_nvs_principled_proposals.md  — literature-informed proposals + failed regs
- v4_nvs_scale_out.md       — 7-scene sweep + per-scene ceiling analysis

Current state:
- Best NVS result so far: bracket-anchor regularizer (in train_gaussian_nvs.py)
  gives +0.04 mean test cc and closes ~47% of the gap to the linear-interpolation
  ceiling on 4 of 7 scenes. The regularizer code already exists and is working.
- Multi-frame NVS has a structural ceiling of ~0.73 on well-sampled scenes,
  bounded by linear interpolation of bracketing train GTs. The 0.11 gap from
  0.73 to train cc (0.84) is non-linear pose response that can only be learned
  from direct test-frame supervision (not allowed in NVS).

Next task (decided in the last session):

Leverage the 16 slow-time loops within each cascaded radar frame as additional
training data. Each cascaded_frame_*.npy is shape (16, 16, 12, 256) = 16 loops
x 16 RX x 12 TX x 256 range samples. The existing pipeline uses only loop 0
(arr[0]) via adc_to_ra_complex, discarding the other 15. We verified loop 0 and
loop 15 differ by ~37% in |RA| magnitude and that the frame duration is ~7.87 ms,
so each loop carries independent information (independent ADC noise + sub-mm
physical pose shift at ~100 mm/s radar velocity).

The plan is a two-stage validation:

STAGE 1 — Loop-level NVS within a single frame (sanity check):
  - Pick one frame (e.g., seq_0_frame_135 frame 131)
  - Process each of its 16 loops separately into a |RA| image
  - Define a train/test split over the 16 loops (e.g., train on even loops,
    test on odd loops)
  - Verify that the model can fit the train loops and predict the test loops
    with high cart_corr (sub-mm pose shifts should be trivial for a well-posed
    model)
  - If this works, we know the pipeline/methodology is sound

STAGE 2 — Use all 16 loops per frame as augmentation for multi-frame NVS:
  - Run Scenario 6/2 (train {0,1,3,5,7,8}, test {2,6}, drop idx 4) on
    seq_0_frame_135
  - Instead of using loop 0 of each train frame, use all 16 loops of each
    train frame as independent training examples (6 * 16 = 96 examples
    per iter vs the current 6)
  - Compare test cc to the baseline 6/2 result (~0.55) and to the bracket-
    anchor result (~0.61)

Hypothesis: using 16x more data per frame will give the model more constraint
at each train pose, reducing the "gauge ambiguity" that causes the optimizer
to carve discontinuities at in-between poses.

Constraints to respect (saved in memory/):
- NVS is interpolation only; never extrapolate to trajectory edges
- Max 500 iterations per run
- Don't use 100x smaller LR as a workaround
- Don't chain warm-start retraining — iterative training is a hack
- Frame 4 (idx 4) is the "reference" frame and is alignment-broken in 4/7 scenes
- Default LRs: mat_lr=0.01, rot_lr=5e-3
- pcl.npy has 7 columns including intensity at col 6 (already wired into init)

Key files:
- mm25DGS_v4/train_gaussian_nvs.py        — NVS trainer with bracket-anchor reg
- mmir/data/ra_utils.py                   — adc_to_ra_complex (needs to support
                                              per-loop processing)
- data/seq_*/radar/cascaded_frame_*.npy   — 16 loops per file
- data/seq_*/configs/cascaded_frame_*.json — timing params (rampEndTime, etc.)

Environment: conda env `mmir`, Python at /home/adnan/.conda/envs/mmir/bin/python.
GPU: 2x RTX 4090 (CUDA_VISIBLE_DEVICES=0 usually, check nvidia-smi).

First step: check how adc_to_ra_complex currently handles the (TX, RX, K, 2)
input and what a minimal modification looks like to take an optional loop_idx
argument. Then design the STAGE 1 experiment (loop-level NVS on a single frame)
and report the results before moving to STAGE 2.

Please proceed principled, not iterative. If a regularizer fails, diagnose why
(check against the "what we ruled out" table in v4_nvs_status_report.md) and
propose something architecturally different rather than tweaking hyperparameters.
```

---

## Notes for the next session

- The 7-scene per-frame all-in ceiling table in
  `v4_nvs_status_report.md` Section 2 is being filled in by a
  background sweep started at the end of the last session. Check
  `/tmp/nvs_all_scenes_allin.log` or
  `/tmp/nvs_allin_per_scene.json` for the output, and update the
  table in the status report if it hasn't been written yet.
- The chirp-diff visualization is at
  `mm25DGS_v4/output_nvs/chirp_diff_analysis/chirp_diff_cart.png`.
  The per-loop trajectory is at
  `mm25DGS_v4/output_nvs/chirp_diff_analysis/per_loop_trajectory.png`.
- The 7-scene regularizer sweep results are in
  `/tmp/nvs_all_scenes_62_results.json` (loaded by the per-scene
  ceiling analysis script).
- Bracket anchor hyperparameters that worked: `bracket_anchor=True,
  bracket_anchor_lambda=0.05, bracket_target='model'`,
  non-detached. Replicated 3× on `seq_0_frame_135` with σ=±0.003.
- All 35 commits through `f42e933` have been pushed to `origin/pt`.
