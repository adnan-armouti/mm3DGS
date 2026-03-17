"""
Free-Space Diffraction BSDF for mmWave Radar.

Implements Steinberg et al. "A Free-Space Diffraction BSDF" (SIGGRAPH 2024)
adapted for 77 GHz FMCW radar with complex E-field phasor output.

Single-bounce implementation: diffraction is evaluated at hit points near
diffracting edges and added to the reflection BSDF via energy-conserving
three-way partition (reflection / transmission / diffraction).
"""

from .fsd_config import DiffractionConfig
from .triangle_search import TriangleSpatialHash
from .fsd_aperture import FsdAperture, FsdEdge, construct_aperture
from .fsd_aperture import _jones_reflectance_numpy
from .fsd_bsdf import FsdBSDF, FlatEdgeData, pack_apertures, eval_psihat_batch_numpy, eval_batch, eval_batch_drjit, _jones_reflectance_drjit
# CPU aperture construction is deprecated; GPU path is always used.
# from .fsd_aperture_batch import construct_apertures_batch
from .fsd_aperture_gpu import construct_apertures_batch_gpu, GPUSpatialHashData, build_gpu_spatial_hash
from .fsd_sampling_tables import FsdSamplingTables, get_fsd_tables
from .fsd_sampling import (
    sample_fsd_edge_drjit,
    eval_fsd_pdf_drjit,
    sample_fsd_sir_drjit,
    sample_fsd_direction_drjit,
    psihat2_per_edge_drjit,
    pjhat_drjit,
    world_to_screen_drjit,
    screen_to_world_drjit,
)

__all__ = [
    'DiffractionConfig',
    'TriangleSpatialHash',
    'FsdAperture',
    'FsdEdge',
    'construct_aperture',
    'FsdBSDF',
    'FlatEdgeData',
    'pack_apertures',
    'eval_psihat_batch_numpy',
    'eval_batch',
    'eval_batch_drjit',
    '_jones_reflectance_numpy',
    '_jones_reflectance_drjit',
    # 'construct_apertures_batch',  # CPU path deprecated; GPU always used
    'construct_apertures_batch_gpu',
    'GPUSpatialHashData',
    'build_gpu_spatial_hash',
    # FSD importance sampling
    'FsdSamplingTables',
    'get_fsd_tables',
    'sample_fsd_edge_drjit',
    'eval_fsd_pdf_drjit',
    'sample_fsd_sir_drjit',
    'sample_fsd_direction_drjit',
    'psihat2_per_edge_drjit',
    'pjhat_drjit',
    'world_to_screen_drjit',
    'screen_to_world_drjit',
]
