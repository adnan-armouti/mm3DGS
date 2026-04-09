# mm25DGS Progress Summary

## Best Result So Far

**Mean cart_corr = 0.753** (7 common scenes) with:
- Constant LR (no decay) 
- 1-way path loss (1/d_tx² matching mmIR)
- Jones BSDF (full KA+SPM, all 6 ITU params)
- Linear RA magnitude loss (no log)
- PCA normals from LiDAR
- 25K Gaussians, 500 iterations, no densification

## Progression

| Change | Mean (7) | Delta |
|--------|----------|-------|
| Baseline (LR decay + log loss + wrong GT frame) | 0.196 | — |
| Fix GT frame | 0.196 | +0.000 |
| Fix log → linear loss | 0.663 | +0.467 |
| Fix LR decay → constant | 0.745 | +0.082 |
| Fix 2-way → 1-way path loss | 0.753 | +0.008 |
| Add mesh normals (regressed some scenes) | 0.725 | -0.028 |

## Target: mmIR single-bounce = 0.936

## Remaining gap: 0.753 → 0.919 = 0.166

## Key remaining differences vs mmIR

1. **Representation**: 25K point Gaussians vs 200K-triangle continuous mesh
2. **Normal accuracy**: PCA normals (mean 38.6° error) vs Poisson mesh normals
3. **No visibility/occlusion testing**: ghost contributions from behind walls
4. **Monostatic BSDF**: one evaluation per Gaussian vs per-TX-RX in mmIR
5. **No MC normalization**: mmIR's reservoir sampling concentrates on important hits
