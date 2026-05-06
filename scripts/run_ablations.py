#!/usr/bin/env python
"""Orchestrator for the NeurIPS supplement ablation suite (Tier 1 + Tier 2).

Architecture
------------
* Defines every (axis, config) row of Supplement Table 8 as a Python dict.
* For each config × scene, builds a ``mm25DGS_v5_v4.train_frame_nvs``
  command and a unique --output_dir under
  ``mm25DGS_v5_v4/output_ablations/<tier>/<axis>/<config>/<scene>/``.
* Maintains 2 GPU slots (CUDA_VISIBLE_DEVICES=0 and =1). When both are
  busy, polls the running ``subprocess.Popen`` handles every 5 s and
  starts the next queued job as soon as a slot frees.
* After all 6 scenes of a config finish, runs ``mmir.evaluation.eval_crp_adc``
  with --ours_dir + --run_tag overrides on that config's output dir and
  re-emits ``latex/.../tables/ablations.tex`` so partial-state visibility
  is preserved if the suite is interrupted.
* Skips configs that already have a populated ``results.json`` for all 6
  scenes (resume-friendly).

Usage
-----
    python scripts/run_ablations.py \\
        --tier1 --tier2 \\
        [--axes axis_dir [axis_dir ...]] \\
        [--scenes scene [scene ...]] \\
        [--dry_run]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import List, Optional, Tuple


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = "/home/adnan/.conda/envs/mmir/bin/python"
ABLATIONS_ROOT = os.path.join(PROJECT_ROOT, "mm25DGS_v5_v4", "output_ablations")
CRP_ADC_ROOT = os.path.join(PROJECT_ROOT, "output", "crp_adc_eval_ablations")
LOG_ROOT = os.path.join(ABLATIONS_ROOT, "_logs")
ORCHESTRATOR_LOG = os.path.join(LOG_ROOT, "orchestrator.log")
GPU_IDS = (0, 1)
POLL_S = 5
PROMOTE_REENTRY_S = 1


# ---------------------------------------------------------------------------
# Logging — own file, line-buffered, also echo to stdout. Avoids the
# SIGPIPE-on-print failure mode that killed the first attempt when a tee
# in the launch pipeline died early.
# ---------------------------------------------------------------------------
_log_fp = None


def _log_open():
    global _log_fp
    os.makedirs(LOG_ROOT, exist_ok=True)
    _log_fp = open(ORCHESTRATOR_LOG, "a", buffering=1)  # line-buffered
    _log_fp.write(f"\n# === orchestrator start {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n")


def log(msg: str = ""):
    if _log_fp is not None:
        try:
            _log_fp.write(msg + "\n")
        except Exception:
            pass
    try:
        print(msg, flush=True)
    except (BrokenPipeError, OSError):
        # If stdout is gone (broken pipe), keep going — we have the file.
        pass


# ---------------------------------------------------------------------------
# Scene config (matches eval_crp_adc.PAPER_SCENES_FULL)
# ---------------------------------------------------------------------------

SCENES = [
    ("seq_0_frame_135", 135, [131, 132, 133, 134, 136, 137, 138, 139]),
    ("seq_1_frame_185", 185, [181, 182, 183, 184, 186, 187, 188, 189]),
    ("seq_1_frame_438", 438, [434, 435, 436, 437, 439, 440, 441, 442]),
    ("seq_2_frame_105", 105, [101, 102, 103, 104, 106, 107, 108, 109]),
    ("seq_2_frame_160", 160, [156, 157, 158, 159, 161, 162, 163, 164]),
    ("seq_2_frame_300", 300, [296, 297, 298, 299, 301, 302, 303, 304]),
]


# ---------------------------------------------------------------------------
# Ablation matrix
# ---------------------------------------------------------------------------
# Each entry: (tier, axis_dir, config_name, extra_flags, override_views)
#   tier         : 'tier1' | 'tier2'
#   axis_dir     : sub-dir under <tier>/ — must match figures/generate_ablation_table.py ROWS
#   config_name  : sub-dir under axis_dir/ — must match generator
#   extra_flags  : list of additional CLI flags for train_frame_nvs
#   override_views : optional override for the per-scene train_frames list
#                    (used by Tier-1 axis-2: subsets of the 8 default frames).
#                    Format: integer count (n); we pick the n closest-to-center
#                    train_frames so the test frame is always bracketed.
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Config:
    tier: str
    axis_dir: str
    config_name: str
    extra_flags: List[str]
    override_views: Optional[int] = None
    target_n_override: Optional[int] = None  # None means use default 20000


def _build_configs(skip_axes: Optional[set] = None) -> List[Config]:
    """Build the full Tier 1 + Tier 2 config list.

    Tier 3 is excluded (BSDF MLP deferred).
    """
    skip_axes = skip_axes or set()
    cfgs: List[Config] = []

    # ── Tier 1 axis 1 — point count N ──
    if "tier1/point_count_N" not in skip_axes:
        for N, name in [(2000, "N_2k"), (5000, "N_5k"),
                          (10000, "N_10k"), (50000, "N_50k")]:
            cfgs.append(Config(
                tier="tier1",
                axis_dir="tier1/point_count_N",
                config_name=name,
                extra_flags=["--target_n", str(N)],
                target_n_override=N,
            ))

    # ── Tier 1 axis 2 — number of training views ──
    # 2 / 4 / 6 views; 8 = default, skipped.
    if "tier1/num_train_views" not in skip_axes:
        for n_views in (2, 4, 6):
            cfgs.append(Config(
                tier="tier1",
                axis_dir="tier1/num_train_views",
                config_name=f"views_{n_views}",
                extra_flags=[],            # train_frames computed per scene
                override_views=n_views,
            ))

    # ── Tier 1 axis 3 — adaptive density off ──
    if "tier1/adaptive_density" not in skip_axes:
        cfgs.append(Config(
            tier="tier1",
            axis_dir="tier1/adaptive_density",
            config_name="off",
            extra_flags=["--reg_densify_interval", "0"],
        ))

    # ── Tier 1 axis 4 — LiDAR init stages ──
    # Note: "no_occlusion" is no longer ablated here. The canonical 3DPS
    # recipe disables the Mitsuba ray-cast by default (it net-hurts test
    # |RA| Corr; see ablation result that motivated this change), so the
    # default IS no_occlusion. Re-enable for ablation by passing
    # --enable_occlusion to train_frame_nvs.py.
    if "tier1/lidar_init" not in skip_axes:
        for tag, flag in [
            ("no_cull",            "--no_cull"),
            ("no_cosine_resample", "--no_cosine_resample"),
            ("no_fps",             "--no_fps"),
        ]:
            cfgs.append(Config(
                tier="tier1",
                axis_dir="tier1/lidar_init",
                config_name=tag,
                extra_flags=[flag],
            ))

    # ── Tier 1 axis 5 — MIMO factorization off ──
    # Risk: PyTorch fallback may OOM at N=20k. Try N=20k first; on OOM
    # the runner re-queues at N=5k (handled in the dispatch loop).
    if "tier1/mimo_factorization" not in skip_axes:
        cfgs.append(Config(
            tier="tier1",
            axis_dir="tier1/mimo_factorization",
            config_name="off",
            extra_flags=["--no_mimo_factorization"],
        ))

    # ── Tier 2 axis 6 — PSF kernel L (skip 15 = default) ──
    if "tier2/psf_kernel_L" not in skip_axes:
        for L in (5, 9, 21, 25):
            cfgs.append(Config(
                tier="tier2",
                axis_dir="tier2/psf_kernel_L",
                config_name=f"L_{L}",
                extra_flags=["--psf_spread", str(L)],
            ))

    # ── Tier 2 axis 7 — phase detach off ──
    if "tier2/phase_detach" not in skip_axes:
        cfgs.append(Config(
            tier="tier2",
            axis_dir="tier2/phase_detach",
            config_name="off",
            extra_flags=["--no_phase_detach"],
        ))

    # ── Tier 2 axis 8 — λ_pos sweep (skip 100 = default) ──
    if "tier2/lambda_pos" not in skip_axes:
        for lam, name in [(0.0, "lpos_0"), (1.0, "lpos_1"), (1000.0, "lpos_1000")]:
            cfgs.append(Config(
                tier="tier2",
                axis_dir="tier2/lambda_pos",
                config_name=name,
                extra_flags=["--learn_positions_l2", str(lam)],
            ))

    return cfgs


# ---------------------------------------------------------------------------
# Train-frames subset for the views-count ablation
# ---------------------------------------------------------------------------

def _train_frames_subset(test_frame: int, all_train: List[int], n_views: int) -> List[int]:
    """Pick the n_views closest-to-test-frame train frames, alternating
    sides so the test frame is always bracketed (NVS interpolation, not
    extrapolation — see feedback_nvs_interpolation_only.md).
    """
    assert n_views in (2, 4, 6)
    left = sorted([f for f in all_train if f < test_frame], reverse=True)   # F-1, F-2, ...
    right = sorted([f for f in all_train if f > test_frame])                # F+1, F+2, ...
    pick = []
    while len(pick) < n_views:
        if left and (len(pick) % 2 == 0 or not right):
            pick.append(left.pop(0))
        elif right:
            pick.append(right.pop(0))
        else:
            break
    return sorted(pick)


# ---------------------------------------------------------------------------
# Job spec
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Job:
    cfg: Config
    scene: str
    test_frame: int
    train_frames: List[int]
    output_dir: str
    cmd: List[str]


def _scene_output_dir(cfg: Config, scene: str) -> str:
    return os.path.join(ABLATIONS_ROOT, cfg.axis_dir, cfg.config_name, scene)


def _build_job(cfg: Config, scene: str, test_frame: int,
                train_frames: List[int], target_n_fallback: Optional[int] = None) -> Job:
    """Build the train_frame_nvs subprocess command for one (cfg, scene)."""
    if cfg.override_views is not None:
        train_frames = _train_frames_subset(test_frame, train_frames, cfg.override_views)
    out = _scene_output_dir(cfg, scene)
    extra = list(cfg.extra_flags)
    # OOM fallback: if a previous attempt at default N OOMed for this
    # config, target_n_fallback overrides --target_n on retry.
    if target_n_fallback is not None and "--target_n" not in extra:
        extra += ["--target_n", str(target_n_fallback)]
    cmd = [
        PYTHON, "-m", "mm25DGS_v5_v4.train_frame_nvs",
        "--scene", scene,
        "--test_frame", str(test_frame),
        "--train_frames", ",".join(str(f) for f in train_frames),
        "--train_loops", "0",
        "--iters", "500",
        "--output_dir", out,
    ] + extra
    return Job(cfg=cfg, scene=scene, test_frame=test_frame,
                train_frames=train_frames, output_dir=out, cmd=cmd)


def _is_done(job: Job) -> bool:
    """A job is considered done if results.json + metrics_train.json + the
    rendered_test_rp_complex.npy file (CRP/ADC eval input) all exist."""
    rp = os.path.join(job.output_dir, "results.json")
    mt = os.path.join(job.output_dir, "metrics_train.json")
    cx = os.path.join(job.output_dir, "rendered_test_rp_complex.npy")
    return all(os.path.isfile(p) for p in (rp, mt, cx))


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Slot:
    gpu_id: int
    proc: Optional[subprocess.Popen] = None
    job: Optional[Job] = None
    log_fp: Optional[object] = None
    started_at: float = 0.0


def _start_job(slot: Slot, job: Job, dry_run: bool = False) -> None:
    os.makedirs(job.output_dir, exist_ok=True)
    os.makedirs(LOG_ROOT, exist_ok=True)
    log_p = os.path.join(LOG_ROOT,
                          f"{job.cfg.axis_dir.replace('/', '_')}__"
                          f"{job.cfg.config_name}__{job.scene}.log")
    log(f"  [GPU {slot.gpu_id}] START {job.cfg.axis_dir}/{job.cfg.config_name} "
          f"× {job.scene}  → {os.path.relpath(job.output_dir, PROJECT_ROOT)}")
    if dry_run:
        log(f"    cmd: {' '.join(job.cmd)}")
        return
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(slot.gpu_id)
    log_fp = open(log_p, "w")
    log_fp.write(f"# {' '.join(job.cmd)}\n")
    log_fp.write(f"# CUDA_VISIBLE_DEVICES={slot.gpu_id}\n")
    log_fp.write(f"# started {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    log_fp.flush()
    # `start_new_session=True` puts the child in its own process group
    # so it survives an orchestrator crash (the previous run died from a
    # SIGPIPE through a broken stdout pipe and took its grandchildren
    # down with it).
    slot.proc = subprocess.Popen(
        job.cmd, env=env, stdout=log_fp, stderr=subprocess.STDOUT,
        start_new_session=True)
    slot.job = job
    slot.log_fp = log_fp
    slot.started_at = time.time()


def _check_slot(slot: Slot) -> Tuple[bool, Optional[str]]:
    """Return (finished, error_msg). error_msg is None on clean exit."""
    if slot.proc is None:
        return True, None
    rc = slot.proc.poll()
    if rc is None:
        return False, None
    elapsed = time.time() - slot.started_at
    err = None
    if rc != 0:
        err = f"non-zero exit ({rc})"
    if slot.log_fp is not None:
        slot.log_fp.write(f"# finished rc={rc} elapsed={elapsed:.1f}s\n")
        slot.log_fp.close()
    if err:
        # Tail the log for OOM / segfault detection
        try:
            with open(slot.log_fp.name) as f:
                tail = f.read()[-2000:]
            if "CUDA out of memory" in tail or "OutOfMemoryError" in tail:
                err = "OOM"
        except Exception:
            pass
    log(f"  [GPU {slot.gpu_id}] DONE  {slot.job.cfg.axis_dir}/{slot.job.cfg.config_name} "
          f"× {slot.job.scene}  ({elapsed:.0f}s)" +
          (f"  ERROR: {err}" if err else ""))
    slot.proc = None
    slot.job = None
    slot.log_fp = None
    return True, err


# ---------------------------------------------------------------------------
# Per-config CRP/ADC eval + table refresh
# ---------------------------------------------------------------------------

def _run_crp_adc_eval(cfg: Config) -> bool:
    config_dir = os.path.join(ABLATIONS_ROOT, cfg.axis_dir, cfg.config_name)
    crp_out = os.path.join(CRP_ADC_ROOT, cfg.axis_dir, cfg.config_name)
    os.makedirs(crp_out, exist_ok=True)
    cmd = [
        PYTHON, "-m", "mmir.evaluation.eval_crp_adc",
        "--ours_dir", config_dir,
        "--run_tag", "{scene}",
        "--output_dir", crp_out,
    ]
    log_p = os.path.join(LOG_ROOT,
                          f"crp_adc__{cfg.axis_dir.replace('/', '_')}__"
                          f"{cfg.config_name}.log")
    log(f"  [eval] CRP/ADC {cfg.axis_dir}/{cfg.config_name} → "
          f"{os.path.relpath(crp_out, PROJECT_ROOT)}")
    with open(log_p, "w") as f:
        f.write(f"# {' '.join(cmd)}\n")
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}).returncode
    return rc == 0


def _refresh_table() -> None:
    cmd = [PYTHON, "-m", "figures.generate_ablation_table"]
    rc = subprocess.run(cmd, capture_output=True, text=True).returncode
    if rc != 0:
        log(f"  [warn] table refresh failed (rc={rc})")
    else:
        log(f"  [refresh] ablations.tex updated")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier1", action="store_true", help="Include Tier 1 axes")
    ap.add_argument("--tier2", action="store_true", help="Include Tier 2 axes")
    ap.add_argument("--axes", nargs="+", default=None,
                    help="Restrict to a subset of axis_dir values.")
    ap.add_argument("--scenes", nargs="+", default=None,
                    help="Restrict to a subset of scene names.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print job plan without launching subprocesses.")
    args = ap.parse_args()

    _log_open()

    if not (args.tier1 or args.tier2):
        # default: both
        args.tier1 = args.tier2 = True

    skip_axes = set()
    if not args.tier1:
        skip_axes.update({c.axis_dir for c in _build_configs() if c.tier == "tier1"})
    if not args.tier2:
        skip_axes.update({c.axis_dir for c in _build_configs() if c.tier == "tier2"})
    cfgs = _build_configs(skip_axes=skip_axes)
    if args.axes is not None:
        wanted = set(args.axes)
        cfgs = [c for c in cfgs if c.axis_dir in wanted]

    scenes_subset = args.scenes
    if scenes_subset is not None:
        scene_specs = [(s, F, tf) for (s, F, tf) in SCENES if s in scenes_subset]
    else:
        scene_specs = list(SCENES)

    log(f"[plan] {len(cfgs)} configs × {len(scene_specs)} scenes = "
          f"{len(cfgs) * len(scene_specs)} runs.")
    log(f"[plan] axes:")
    last_axis = None
    for c in cfgs:
        if c.axis_dir != last_axis:
            log(f"  {c.axis_dir}:")
            last_axis = c.axis_dir
        log(f"    - {c.config_name}  flags={c.extra_flags}"
              + (f"  views={c.override_views}" if c.override_views else ""))

    if args.dry_run:
        return

    # Per-config: build all 6 scene jobs, skip done, dispatch, eval, refresh.
    # Top-level try wraps the whole sweep so a crash in any one config
    # doesn't kill the others (we log + continue).
    slots = [Slot(gpu_id=g) for g in GPU_IDS]
    overall_t0 = time.time()
    try:
      for ci, cfg in enumerate(cfgs):
        try:
            cfg_t0 = time.time()
            log(f"\n[{ci+1}/{len(cfgs)}] === {cfg.axis_dir}/{cfg.config_name} ===")
            # Build all jobs for this config
            jobs: List[Job] = []
            for scene, F, tf in scene_specs:
                jobs.append(_build_job(cfg, scene, F, tf))

            # Dispatch loop
            retry_with_smaller_n = False
            pending = [j for j in jobs if not _is_done(j)]
            if not pending:
                log("  all 6 scenes already done; skipping dispatch")
            while pending or any(s.proc is not None for s in slots):
                # Start new jobs in any free slot
                for slot in slots:
                    if slot.proc is None and pending:
                        j = pending.pop(0)
                        _start_job(slot, j, dry_run=False)
                        time.sleep(PROMOTE_REENTRY_S)  # avoid two near-simultaneous Mitsuba inits
                # Poll
                time.sleep(POLL_S)
                for slot in slots:
                    done, err = _check_slot(slot)
                    if done and err == "OOM" and not retry_with_smaller_n:
                        retry_with_smaller_n = True
                        log(f"  [oom] re-queueing {cfg.axis_dir}/{cfg.config_name} "
                              f"at N=5000 fallback")
                        pending = [_build_job(cfg, s, F, tf, target_n_fallback=5000)
                                   for (s, F, tf) in scene_specs
                                   if not _is_done(_build_job(cfg, s, F, tf,
                                                              target_n_fallback=5000))]

            # Per-config CRP/ADC eval + table refresh
            n_done = sum(1 for j in jobs if _is_done(j))
            if n_done > 0:
                _run_crp_adc_eval(cfg)
                _refresh_table()
            cfg_dt = time.time() - cfg_t0
            log(f"  {cfg.axis_dir}/{cfg.config_name} complete  "
                  f"({n_done}/{len(jobs)} scenes, {cfg_dt:.0f}s)")
            log(f"  cumulative wall time: {(time.time() - overall_t0)/3600:.2f} h")
        except Exception as e:
            import traceback
            log(f"  [ERROR] {cfg.axis_dir}/{cfg.config_name} crashed: {e}")
            log(traceback.format_exc())
            # Reap any leftover slots so the next config doesn't inherit them.
            for slot in slots:
                if slot.proc is not None:
                    try:
                        slot.proc.terminate()
                    except Exception:
                        pass
                    slot.proc = None
                    slot.job = None
                    slot.log_fp = None
            continue
    except KeyboardInterrupt:
        log("\n[interrupted] user requested stop; reaping in-flight slots")
        for slot in slots:
            if slot.proc is not None:
                try:
                    slot.proc.terminate()
                except Exception:
                    pass
        return

    log(f"\n[done] all configs processed in {(time.time() - overall_t0)/3600:.2f} wall-h")


if __name__ == "__main__":
    main()
