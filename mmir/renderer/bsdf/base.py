"""
Abstract base class for BSDF implementations.

Standard Computer Graphics BSDF Interface:
- eval_f(wo, wi, n, material): Evaluate f(wo, wi) - NO cosine
- eval_f_cos(wo, wi, n, material): Evaluate f(wo, wi) * |cos(theta_i)|
- pdf_wi(wo, wi, n, material): PDF of sampling wi given wo
- sample_f(wo, n, material, samples): Sample wi direction
- value(wo, wi, n, material): eval_f / pdf_wi (importance sampling weight)

Coordinate Conventions:
- wo: Outgoing direction (toward receiver/camera) - world space
- wi: Incident direction (toward light/transmitter) - world space
- n: Surface normal - world space
- All vectors normalized

Material Parameters (rho):
- albedo (rho.x): diffuse reflectance
- roughness (rho.y): alpha for GGX
- metallic (rho.z): 0=dielectric, 1=metal

Critical Convention Rules (MUST BE ENFORCED):

Rule A: Single BSDF Entry Point
    ALL integrator paths must call bsdf.eval_f_cos() — no exceptions.
    No "diffuse-only fast paths" that bypass the BSDF class.

Rule B: eval_f() Returns NO Cosine
    eval_f() returns f(wo, wi) with NO cosine factor.
    eval_f_cos() = eval_f() * max(dot(n, wi), 0)

Rule C: pdf_wi() / value() — Use Only When BSDF-Sampled
    If wi came from pattern/reservoir/NEE sampling, use that sampler's pdf.
    Current single-bounce renderer uses pattern sampling, NOT BSDF sampling.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union
import numpy as np


class BSDFBase(ABC):
    """Abstract base class for BSDF implementations."""

    # Metal BRDF calibration for 77 GHz radar
    F0_DIELECTRIC = 0.04  # Fresnel at normal incidence for dielectrics
    F0_METAL = 0.95       # Fresnel at normal incidence for metals at 77 GHz

    @abstractmethod
    def eval_f(
        self,
        wo: Union[np.ndarray, 'mi.Vector3f'],
        wi: Union[np.ndarray, 'mi.Vector3f'],
        n: Union[np.ndarray, 'mi.Vector3f'],
        albedo: Union[np.ndarray, 'mi.Float'],
        roughness: Union[np.ndarray, 'mi.Float'],
        metallic: Union[np.ndarray, 'mi.Float'],
    ) -> Union[np.ndarray, 'mi.Float']:
        """
        Evaluate BSDF: f(wo, wi).

        CRITICAL: Returns f(wo,wi) with NO cosine factor.

        Returns:
            BSDF value (scalar per path)
        """
        pass

    @abstractmethod
    def eval_f_cos(
        self,
        wo: Union[np.ndarray, 'mi.Vector3f'],
        wi: Union[np.ndarray, 'mi.Vector3f'],
        n: Union[np.ndarray, 'mi.Vector3f'],
        albedo: Union[np.ndarray, 'mi.Float'],
        roughness: Union[np.ndarray, 'mi.Float'],
        metallic: Union[np.ndarray, 'mi.Float'],
    ) -> Union[np.ndarray, 'mi.Float']:
        """
        Evaluate BSDF × |cos(theta_i)|.

        For RX-centric radar MC estimator:
        - cos(theta_out) absorbed by solid-angle change of variables
        - cos(theta_in) is explicit factor (c_t in estimator)

        Returns:
            f(wo, wi) × |cos(theta_i)|
        """
        pass

    @abstractmethod
    def pdf_wi(
        self,
        wo: Union[np.ndarray, 'mi.Vector3f'],
        wi: Union[np.ndarray, 'mi.Vector3f'],
        n: Union[np.ndarray, 'mi.Vector3f'],
        roughness: Union[np.ndarray, 'mi.Float'],
    ) -> Union[np.ndarray, 'mi.Float']:
        """
        PDF of sampling incident direction wi given outgoing direction wo.

        WARNING: Only use this when wi was sampled via BSDF.sample_f()!
        If wi came from pattern/reservoir/NEE sampling, use that sampler's
        pdf instead. Using BSDF pdf for non-BSDF-sampled directions gives
        WRONG importance weights.

        Current single-bounce renderer uses pattern/reservoir sampling,
        so this method should NOT be used in the integrator.
        Reserved for future multi-bounce path tracing with BSDF sampling.

        Returns:
            p(wi | wo)
        """
        pass

    @abstractmethod
    def sample_f(
        self,
        wo: Union[np.ndarray, 'mi.Vector3f'],
        n: Union[np.ndarray, 'mi.Vector3f'],
        albedo: Union[np.ndarray, 'mi.Float'],
        roughness: Union[np.ndarray, 'mi.Float'],
        metallic: Union[np.ndarray, 'mi.Float'],
        samples: Union[np.ndarray, 'mi.Point2f'],
    ) -> Tuple[
        Union[np.ndarray, 'mi.Vector3f'],  # wi
        Union[np.ndarray, 'mi.Float'],      # f_cos (f × cos_theta_in)
        Union[np.ndarray, 'mi.Float'],      # pdf
        Union[np.ndarray, 'mi.Bool'],       # is_delta
    ]:
        """
        Sample incident direction wi given outgoing direction wo.

        Returns:
            (wi, f_cos, pdf, is_delta):
            - wi: Sampled incident direction
            - f_cos: BSDF value × |cos(theta_in)| (the throughput weight)
            - pdf: PDF of sampling this direction (solid angle measure)
            - is_delta: True if sample from delta distribution (perfect mirror)

        Returning f_cos directly (instead of just pdf) simplifies path throughput:
            throughput *= f_cos / pdf  # No separate eval_f call needed

        For delta distributions (is_delta=True):
        - pdf is formally infinite (represented as 1.0 for convention)
        - MIS weight should be 1.0 (no competing strategy can sample deltas)
        - The returned wi is the only valid direction for that path
        - f_cos already includes the "infinite" cancellation

        Not used in current single-bounce renderer (uses pattern sampling).
        Reserved for future multi-bounce path tracing.
        """
        pass

    def value(
        self,
        wo: Union[np.ndarray, 'mi.Vector3f'],
        wi: Union[np.ndarray, 'mi.Vector3f'],
        n: Union[np.ndarray, 'mi.Vector3f'],
        albedo: Union[np.ndarray, 'mi.Float'],
        roughness: Union[np.ndarray, 'mi.Float'],
        metallic: Union[np.ndarray, 'mi.Float'],
    ) -> Union[np.ndarray, 'mi.Float']:
        """
        Importance sampling weight: eval_f / pdf_wi.

        WARNING: Only use this when wi was sampled via BSDF.sample_f()!
        For the current single-bounce renderer, wi comes from pattern/reservoir
        sampling — DO NOT use this method. Instead:

            # CORRECT for single-bounce renderer:
            brdf_weight = bsdf.eval_f_cos(wo, wi, n, ...)
            final_weight = brdf_weight / sampler_pdf  # Use sampler's pdf

            # WRONG — wi wasn't BSDF-sampled:
            weight = bsdf.value(wo, wi, n, ...)  # Divides by wrong pdf!

        Reserved for future multi-bounce path tracing with BSDF sampling.

        Default implementation; subclasses may override for efficiency.
        """
        import drjit as dr
        f = self.eval_f(wo, wi, n, albedo, roughness, metallic)
        pdf = self.pdf_wi(wo, wi, n, roughness)
        # Handle division by zero
        return f / np.maximum(pdf, 1e-10) if isinstance(f, np.ndarray) else f / dr.maximum(pdf, 1e-10)


__all__ = ['BSDFBase']
