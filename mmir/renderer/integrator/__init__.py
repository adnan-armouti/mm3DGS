"""
SBR Integrator package for FMCW ADC signal synthesis.

Provides the SBRIntegratorRef class with methods for:
- Forward rendering (Phase A)
- Two-phase differentiable rendering (Phase A cache → Phase B AD)
- End-to-end differentiable multibounce rendering

Methods are defined in separate modules and attached to the class here.
"""

# Standalone dataclasses and utilities (no class dependencies)
from .cached_geometry import ADCResult, ADCComponentResult, CachedGeometry
from .math_primitives import diff_ray_triangle_intersect, get_ad_triangle_vertices
from .mixture_sampling import _sample_cosine_hemisphere_drjit, _sample_mixture_bsdf_drjit

# Core class definition (includes __init__, _precompute_radar_constants, etc.)
from .core import SBRIntegratorRef

# Import method implementations
from .synthesis_forward import (
    synthesize_vectorized,
    synthesize_with_patterns,
    synthesize_shared_hits_vectorized,
)
from .synthesis_e2e import (
    synthesize_end_to_end,
    synthesize_end_to_end_multibounce,
    _synthesize_specular_e2e,
    _construct_bounce_apertures,
    _compute_diffraction_e2e,
)
from .synthesis_differentiable import (
    synthesize_differentiable,
    _recompute_hit_positions,
    _recompute_normals,
    _recompute_antenna_gains,
    _recompute_specular_antenna_gains,
    _recompute_specular_positions_ift,
    _recompute_beta_differentiable,
    _synthesize_specular_chains_e2e,
    _synthesize_specular_differentiable,
    _synthesize_specular_paths,
)

# Attach forward synthesis methods
SBRIntegratorRef.synthesize_vectorized = synthesize_vectorized
SBRIntegratorRef.synthesize_with_patterns = synthesize_with_patterns
SBRIntegratorRef.synthesize_shared_hits_vectorized = synthesize_shared_hits_vectorized

# Attach end-to-end methods
SBRIntegratorRef.synthesize_end_to_end = synthesize_end_to_end
SBRIntegratorRef.synthesize_end_to_end_multibounce = synthesize_end_to_end_multibounce
SBRIntegratorRef._synthesize_specular_e2e = _synthesize_specular_e2e
SBRIntegratorRef._construct_bounce_apertures = _construct_bounce_apertures
SBRIntegratorRef._compute_diffraction_e2e = _compute_diffraction_e2e

# Attach differentiable synthesis methods
SBRIntegratorRef.synthesize_differentiable = synthesize_differentiable
SBRIntegratorRef._recompute_hit_positions = _recompute_hit_positions
SBRIntegratorRef._recompute_normals = _recompute_normals
SBRIntegratorRef._recompute_antenna_gains = _recompute_antenna_gains
SBRIntegratorRef._recompute_specular_antenna_gains = _recompute_specular_antenna_gains
SBRIntegratorRef._recompute_specular_positions_ift = _recompute_specular_positions_ift
SBRIntegratorRef._recompute_beta_differentiable = _recompute_beta_differentiable
SBRIntegratorRef._synthesize_specular_chains_e2e = _synthesize_specular_chains_e2e
SBRIntegratorRef._synthesize_specular_differentiable = _synthesize_specular_differentiable
SBRIntegratorRef._synthesize_specular_paths = _synthesize_specular_paths

__all__ = [
    'SBRIntegratorRef',
    'ADCResult',
    'ADCComponentResult',
    'CachedGeometry',
    'diff_ray_triangle_intersect',
    'get_ad_triangle_vertices',
]
