# Solid Angle Voronoi Weighting for Gaussians

## The insight

mmIR doesn't need areas because each MC sample represents `1/(pdf × N)` steradians of solid angle on the RX hemisphere. The samples are **importance-weighted by the sampling distribution**, not by geometric area.

Our Gaussians use `A × cos_o / d_rx²` as the solid angle weight. This is the geometric solid angle of the surfel — correct if the surfels perfectly tile the scene surface. But they don't: mesh vertices and LiDAR FPS points have non-uniform density, creating gaps and overlaps.

## The fix: assign each Gaussian its Voronoi solid angle on the RX hemisphere

For each RX element, project all visible Gaussians onto the unit sphere centered at RX. Each Gaussian occupies a direction `ω_i = normalize(μ_i - p_rx)`. The Voronoi solid angle of Gaussian i is the area of its Voronoi cell on the unit sphere — the set of directions closer to `ω_i` than to any other Gaussian.

This is the deterministic equivalent of the MC `1/(pdf × N)` weight: it describes how much of the RX hemisphere each Gaussian "represents."

## Practical computation

### Option A: k-NN angular distance approximation

For each RX element r and each Gaussian i:
1. Compute the direction `ω_i = normalize(μ_i - p_rx_r)` (unit sphere projection)
2. Find the k nearest neighbors of ω_i among the other projected Gaussians
3. The Voronoi cell solid angle ≈ `π × θ_nn²` where `θ_nn` is the angular distance to the nearest neighbor

This approximates each Voronoi cell as a spherical cap with radius equal to half the nearest-neighbor angular distance.

**Cost**: one k-NN search per RX element (16 total), each over the visible Gaussian set.
**Computed once at init** (not per iteration).

### Option B: Analytic solid angle from Gaussian scales

Each Gaussian has scales (s₁, s₂) defining its spatial extent. The projected solid angle on the RX hemisphere is:

```
dΩ_i(r) = (π × s₁ × s₂ × |cos θ_rx|) / d_rx²
```

This is the current `A × cos_o / d_rx²` formula but using `A = π × s₁ × s₂` (the Gaussian's area from its scales) instead of the mesh face area.

**Problem**: this only works if the Gaussians perfectly tile the surface. If there are gaps, the sum of all `dΩ_i` will be less than the total visible hemisphere. If there are overlaps, the sum exceeds the hemisphere.

### Option C: Normalized solid angles (hemisphere conservation)

Compute Option B's solid angles, then normalize so they sum to the correct total hemisphere solid angle:

```
dΩ_i_raw = π × s₁ × s₂ × |cos θ_rx| / d_rx²
dΩ_i_normalized = dΩ_i_raw × (Ω_total / Σ_j dΩ_j_raw)
```

where `Ω_total = π` (half-hemisphere, since we only see the forward-facing portion) or more precisely, the solid angle of the visible scene as seen from the RX.

This ensures the total contribution matches the physical constraint, regardless of gaps or overlaps in the Gaussian coverage.

**Cost**: one summation per RX element per iteration (cheap).

### Option D: Replace geometric weights with MC weights directly

The most direct approach: run the reservoir sampler once, get the MC weights `1/(pdf × N)` for each hit, then transfer these weights to the nearest Gaussians.

For each reservoir hit j with weight `w_mc_j = 1/(pdf_j × N_attempted)`:
1. Find the nearest Gaussian i to hit j
2. Accumulate: `mc_weight[i] += w_mc_j`

The resulting `mc_weight[i]` is the total MC weight contributed by all hits near Gaussian i. This IS the correct solid angle that Gaussian i should represent.

**This is different from Option 2B (hit frequency)** because it uses the actual MC weight `1/(pdf × N)`, not just the count. Hit frequency doesn't account for the PDF variation — hits in high-PDF regions (close to the RX boresight) get smaller MC weights than hits in low-PDF regions (at the hemisphere edge).

**Cost**: one reservoir sampler call at init + KDTree query.

## Recommendation

**Option D** is the most principled: it directly transfers the MC integration framework from Stage B to the Gaussian representation. Each Gaussian inherits the total MC weight of the reservoir hits in its neighborhood.

The key difference from the failed Option 2B: Option 2B used hit COUNT (proportional to the PDF), which double-counted the geometric correction. Option D uses the MC WEIGHT (which is `1/pdf` — inversely proportional to the PDF), which replaces the geometric correction entirely.

With Option D, the area weight changes from:
```
areas = vertex_area × opacity × cos_o / d_rx²     (current, wrong)
```
to:
```
areas = mc_weight × opacity                         (Option D, matches Stage B)
```

And the path loss changes from bistatic `1/(d_tx² × d_rx²)` back to:
```
path_loss = 1 / d_tx²                               (as in Stage B / mmIR)
```

because the MC weight already absorbs the `1/d_rx²` factor (same reasoning as the original mmIR derivation).

## Implementation

In `train_gaussian.py`, at init after running the reservoir sampler:

```python
# Option D: transfer MC weights from reservoir hits to Gaussians
hit_positions = extract_hit_positions(hits)     # (n_hits, 3)
hit_mc_weights = extract_mc_weights(hits)       # (n_hits,) = 1/(pdf × n_attempted)
gauss_positions = model.positions.detach().cpu().numpy()

tree = KDTree(gauss_positions)
_, nearest_gauss = tree.query(hit_positions)    # (n_hits,) nearest Gaussian per hit

# Sum MC weights per Gaussian
mc_weight = np.bincount(nearest_gauss, weights=hit_mc_weights, minlength=N)
mc_weight_t = torch.from_numpy(mc_weight.astype(np.float32)).to(device)
```

Then in `render_gaussians_factorized`, use `mc_weight × opacity` instead of `vertex_area × opacity`, and switch back to `1/d_tx²` path loss (not bistatic).

This makes the Gaussian renderer algebraically equivalent to Stage B when the Gaussians are at the hit positions. For Gaussians at LiDAR/mesh positions, it approximates Stage B by transferring the MC weighting to the nearest discrete representation points.
