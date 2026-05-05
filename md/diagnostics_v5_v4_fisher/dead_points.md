# Phase A.5 — Why are dead points dead?
Top 10% = "live" (holds ~63% of total ‖∇p L‖); bottom 90% = "dead" (holds ~37%, mostly bottom 50% which is ~3% of total).

## Per-scene spatial diagnostics
| scene | d(dead→live) m | d(live→live) m | d(pool→live) m | live frac of pool within 0.10 m | test cc |
|---|---:|---:|---:|---:|---:|
| seq_0_frame_135 | 1.185 | 0.008 | 1.713 | 7.1% | 0.5747 |
| seq_1_frame_185 | 0.465 | 0.011 | 0.938 | 16.6% | 0.5413 |
| seq_1_frame_438 | 0.342 | 0.008 | 0.640 | 13.8% | 0.6693 |
| seq_2_frame_105 | 1.031 | 0.008 | 1.541 | 8.2% | 0.5572 |
| seq_2_frame_160 | 0.249 | 0.011 | 0.981 | 6.7% | 0.6318 |
| seq_2_frame_300 | 1.339 | 0.008 | 1.844 | 7.3% | 0.4525 |
| **mean** | **0.769** | **0.009** | **1.276** | **10.0%** | **0.5711** |

## Interpretation
- **d(dead→live)** ≪ **d(live→live)**: dead points sit *near* live points → not init's fault, the optimizer isn't letting them learn. Init replacement won't help.
- **d(dead→live)** ≈ **d(live→live)** or larger: dead points are isolated in low-signal regions → init mis-placed them. Better init should help.
- **pool frac within 10 cm**: how much of the LiDAR pool is redundant w.r.t. our live points. High = lots of pool capacity going unused near active regions; low = active regions are sparsely covered by the pool too.
