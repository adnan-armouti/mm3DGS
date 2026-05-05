# Phase A.6 — Were live points already clustered at INIT?

| scene | d_init(live→live) m | d_final(live→live) m | drift_live m | drift_dead m | test cc |
|---|---:|---:|---:|---:|---:|
| seq_0_frame_135 | 0.132 | 0.028 | 3.1202 | 0.0005 | 0.5747 |
| seq_1_frame_185 | 0.071 | 0.033 | 0.0032 | 0.0012 | 0.5413 |
| seq_1_frame_438 | 0.104 | 0.059 | 0.0026 | 0.0009 | 0.6693 |
| seq_2_frame_105 | 0.107 | 0.027 | 1.3814 | 0.0007 | 0.5572 |
| seq_2_frame_160 | 0.112 | 0.070 | 0.0022 | 0.0009 | 0.6318 |
| seq_2_frame_300 | 0.096 | 0.021 | 5.7807 | 0.0007 | 0.4525 |

## Interpretation
- **d_init ≈ d_final**: live points were already clustered at init. FPS over-concentrated them. → Init IS the problem.
- **d_init ≫ d_final**: live points migrated to clusters during training. → Init was OK; the issue is the optimizer collapsing onto hot spots.
- **drift_live ≈ drift_dead ≈ a few mm**: matches the L2 anchor analytical bound (5 mm). Positions barely move. So the configuration we observe is essentially the init configuration, with materials/normals adapted on top.
