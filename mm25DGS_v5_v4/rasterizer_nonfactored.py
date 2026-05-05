"""Non-factored MIMO renderer wrapper for the MIMO-factorization ablation
(Tier-1 axis 5, ``md/ablations_plan.md``).

Background
----------
The default 3DPS forward model is implemented in
:mod:`mm25DGS_v5_v4.rasterizer_factorized` as ``render_factorized``. That
function takes ``use_cuda_kernels=True`` by default, which routes Step-4
(per-(TX, RX) BSDF) and Step-5 (per-(TX, RX, range-bin) Hann-PSF splat)
through fused CUDA kernels that *avoid* materialising the
:math:`(M, n_{tx}, n_{rx})` BSDF tensor and the
:math:`(\\text{spread}, M \\cdot n_{tx} \\cdot n_{rx})` splat tensor. This is the
"quasi-static MIMO factorization" optimisation introduced by commits
``707a3ab`` (BSDF Phase B) and ``b736733`` (range-splat Phase D).

When ``use_cuda_kernels=False`` is passed, the same function falls through
to a pure-PyTorch implementation that *does* materialise those full
tensors and computes the Cook-Torrance KA + SPM lobes for every
:math:`(M, n_{tx}, n_{rx})` triple via einsum. This is the historical
pre-factorisation renderer (preserved as the fallback when the C++
extension is not built or when an exotic ``bsdf_mode``/``disabled_components``
setup forbids the fused path).

This wrapper exposes that fallback as a named entry point so the
ablation table caption can cite a clean, separately-importable function.
The physics it computes is identical to the factorised path; only the
memory footprint and per-iter wall-clock differ. Expect roughly:

  * ~3-10x slower per Adam iteration (no fused Step-4 / Step-5 kernels).
  * ~10x larger renderer working set for the BSDF+splat intermediates;
    at :math:`M=20\\,000`, :math:`n_{tx}=12`, :math:`n_{rx}=16`,
    :math:`\\text{spread}=15` the (spread, M, n_tx, n_rx) range-splat
    contribution tensor alone is ~220 MB at fp32 (vs ~0 in the fused path).

Use only for the MIMO-factorization ablation. The default training
recipe must continue to call ``render_factorized`` directly.
"""

from mm25DGS_v5_v4.rasterizer_factorized import render_factorized


def render_nonfactored(
    positions,
    normals,
    areas,
    raw_materials,
    rast,
    reparameterize_fn,
    detach_phase=True,
    shadow_mask=None,
    bsdf_mode='full',
    disabled_components=None,
    psf_spread=None,
):
    """Non-factored MIMO renderer (ablation entry point).

    Identical signature to ``render_factorized`` except that the fused
    CUDA Step-4 / Step-5 kernels are pinned off, so the function exercises
    the PyTorch path that materialises full :math:`(M, n_{tx}, n_{rx})`
    BSDF tensors and the :math:`(\\text{spread}, M \\cdot n_{tx} \\cdot n_{rx})`
    range-splat contributions tensor.

    Parameters
    ----------
    positions, normals, areas, raw_materials, rast, reparameterize_fn,
    detach_phase, shadow_mask, bsdf_mode, disabled_components :
        Forwarded verbatim to ``render_factorized``.
    psf_spread :
        Optional override for the Hann PSF kernel half-width
        :math:`L`. Forwarded as the ``psf_spread`` kwarg of
        ``render_factorized`` (default :math:`L=15` when ``None``).

    Returns
    -------
    (rp_real, rp_imag) : tuple of torch.Tensor, both :math:`(n_{tx}, n_{rx}, K)`
        Identical to the factorised-renderer output by construction.
    """
    return render_factorized(
        positions, normals, areas, raw_materials, rast,
        reparameterize_fn,
        detach_phase=detach_phase,
        shadow_mask=shadow_mask,
        bsdf_mode=bsdf_mode,
        disabled_components=disabled_components,
        use_cuda_kernels=False,
        psf_spread=psf_spread,
    )
