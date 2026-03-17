"""
Differentiable FMCW Radar Renderer.

Physics-based Monte Carlo renderer for 77 GHz mmWave FMCW radar with:
- RX-centric reservoir sampling
- KA+SPM hybrid BSDF (CSV-BSDF) with Jones polarization
- Specular Manifold Sampling (SMS) for specular paths
- Free-space diffraction (FSD-BSDF, Steinberg et al. SIGGRAPH 2024)
- End-to-end differentiable rendering with multibounce support

Usage:
    from mmir.renderer import FMCWRendererRef, RenderConfigRef

    renderer = FMCWRendererRef.from_files(
        mesh_file="scene.ply",
        config_file="config.json",
        tx_pattern_file="pattern.npy",
        rx_pattern_file="pattern.npy",
    )
    result = renderer.render(seed=42)
"""

# Scene context and logging (inlined from mmir/renderer)
from .scene_context import SceneContext, RenderConfig, RenderLogger

# Core renderer and integrator
from .config import RenderConfigRef, RenderResultRef
from .renderer import FMCWRendererRef
from .integrator import SBRIntegratorRef, ADCResult, ADCComponentResult, CachedGeometry
from .sampler import ReservoirSampler, ReservoirHits, ReservoirHitsDrJit

# BSDF classes
from .bsdf import BSDFBase, BSDFmmWaveScalar, BSDFmmWaveJones
from .bsdf.reparameterization import (
    reparameterize_physics_params,
    inverse_reparameterize,
    reparameterize_physics_params_drjit,
    create_drjit_raw_params,
    LR_SCALES,
)

# Specular path finding
from .specular import (
    ImageMethodRefiner, SpecularPaths,
    SpecularManifoldSampler, SpecularGradInfo,
    MultibounceSpecularChain, MultibounceSpecularGradInfo,
    SpecularDeduplicator, PlaneHasher, hash_fnv1a_uint32,
)

# Materials
from .materials import (
    MaterialParameterization, PerTriangleParameterization,
    PerVertexParameterization, vertex_to_triangle_materials,
    get_itu_properties, get_material_properties, get_wall_preset, list_materials,
    ITU_MATERIAL_PROPERTIES, DEFAULT_ROUGHNESS, DEFAULT_THICKNESS, WALL_PRESETS,
    abcd_multilayer_fresnel, compute_multilayer_energy_gate,
)

# Utilities
from .utils import (
    DrJitAdam,
    perp_stark, to_local, to_global, sample_cosine_hemisphere_concentric,
    safe_normalize, gather_point3f, gather_vector3f,
    euler_to_quaternion, quaternion_rotate, transform_positions, create_pose_params,
)
from .utils.clustering import PatchClusterer, PatchData
from .utils.boundary import BoundaryGradientComputer
from .utils.pattern_sampler import PatternImportanceSampler

# Free-space diffraction BSDF (Steinberg et al. SIGGRAPH 2024)
try:
    from .diffraction import (
        DiffractionConfig,
        TriangleSpatialHash,
        FsdAperture,
        construct_aperture,
        FsdBSDF,
    )
    from .diffraction.triangle_search import build_triangle_hash_from_scene
except ImportError:
    pass

# Backward-compatible import path for material_parameterization
from .materials import parameterization as material_parameterization

__all__ = [
    # Core
    'FMCWRendererRef', 'RenderConfigRef', 'RenderResultRef',
    'SBRIntegratorRef', 'ADCResult', 'ADCComponentResult', 'CachedGeometry',
    'ReservoirSampler', 'ReservoirHits', 'ReservoirHitsDrJit',
    # BSDF
    'BSDFBase', 'BSDFmmWaveScalar', 'BSDFmmWaveJones',
    'reparameterize_physics_params', 'inverse_reparameterize',
    'reparameterize_physics_params_drjit', 'create_drjit_raw_params', 'LR_SCALES',
    # Specular
    'ImageMethodRefiner', 'SpecularPaths',
    'SpecularManifoldSampler', 'SpecularGradInfo',
    'MultibounceSpecularChain', 'MultibounceSpecularGradInfo',
    'SpecularDeduplicator', 'PlaneHasher', 'hash_fnv1a_uint32',
    # Materials
    'MaterialParameterization', 'PerTriangleParameterization',
    'PerVertexParameterization', 'vertex_to_triangle_materials',
    'get_itu_properties', 'get_material_properties', 'get_wall_preset', 'list_materials',
    'ITU_MATERIAL_PROPERTIES', 'DEFAULT_ROUGHNESS', 'DEFAULT_THICKNESS', 'WALL_PRESETS',
    'abcd_multilayer_fresnel', 'compute_multilayer_energy_gate',
    # Utilities
    'DrJitAdam', 'PatchClusterer', 'PatchData',
    'BoundaryGradientComputer', 'PatternImportanceSampler',
    'perp_stark', 'to_local', 'to_global', 'sample_cosine_hemisphere_concentric',
    'safe_normalize', 'gather_point3f', 'gather_vector3f',
    'euler_to_quaternion', 'quaternion_rotate', 'transform_positions', 'create_pose_params',
    # Diffraction
    'DiffractionConfig', 'TriangleSpatialHash', 'FsdAperture', 'construct_aperture', 'FsdBSDF',
    # External reuse
    'SceneContext', 'RenderConfig', 'RenderLogger',
    # Backward compatibility
    'material_parameterization',
]
