# §2.5 Ablation matrix

Single-axis variants; 500 iter, target_n=20000.

## seq_0_frame_135 F=135

| row | |RA| train | |RA| test | |RAD| train | |RAD| test | elapsed (s) |
|---|---:|---:|---:|---:|---:|
| B_baseline | 0.6770 | 0.5990 | 0.4545 | 0.2964 | 294 |
| NORM_MAX_legacy | 0.6675 | 0.6179 | 0.4400 | 0.3063 | 283 |
| M03_multitask | 0.6971 | 0.5826 | 0.4523 | 0.2883 | 326 |
| M10_multitask | 0.7149 | 0.6207 | 0.4458 | 0.2976 | 318 |
| M30_multitask | 0.7150 | 0.6284 | 0.4269 | 0.2761 | 327 |
