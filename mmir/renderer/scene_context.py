"""
Scene context and logging utilities for the FMCW renderer.

Inlined from mmir/renderer/renderer.py to make renderer_final self-contained.
Contains: RenderConfig, SceneContext, RenderLogger
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path
import json

import drjit as dr
import mitsuba as mi
import numpy as np

if TYPE_CHECKING:
    from ..sensor.config import FMCWConfig, TxArray, RxArray
    from ..sensor.element_patterns import AntennaPatternLoader
    from ..sensor.adc_accumulator import FMCWAdcAccumulator
    from .scene_params.vertex_materials import VertexMaterialParams
    from .scene_params.vertex_positions import VertexPositions
    from .scene_params.vertex_normals import VertexNormals
    from .scene_params.edge_data import EdgeDataDr
    from .scene_params.face_data import FaceData


@dataclass
class RenderConfig:
    """
    Configuration for rendering options.

    Controls rendering behavior including depth limits, sampling options,
    and debug toggles.
    """
    # Core rendering parameters
    n_samples: int = 100000
    max_depth: int = 3
    seed: int = 42

    # Feature toggles
    enable_diffraction: bool = True
    enable_mis: bool = True
    enable_antenna_beam_pattern: bool = True
    enable_path_loss: bool = True
    enable_doppler: bool = False

    # Sampling options
    edge_sample_fraction: float = 0.2  # Fraction of rays for edge sampling (0-1)
    sampling_mode: str = "cosine"  # "uniform" or "cosine" hemisphere sampling

    # Russian roulette
    rr_depth: int = 3
    rr_max_prob: float = 0.95

    # Shadow ray epsilon
    shadow_ray_epsilon: float = 1e-4

    # Material model
    use_material_model: bool = True

    # Debug toggles
    verbose: int = 0  # 0=silent, 1=summary, 2=per-bounce, 3=debug
    dump_intermediates: bool = False
    dump_path: Optional[str] = None
    validate_mis_weights: bool = False
    validate_energy: bool = False
    profile_bounces: bool = False


@dataclass
class SceneContext:
    """
    Immutable scene context holding all data required for rendering.

    This bundles together the Mitsuba scene, FMCW configuration,
    antenna arrays, and differentiable parameters into a single
    object that can be passed through the rendering pipeline.

    All fields should be treated as immutable after construction.
    """
    # Mitsuba scene
    scene: 'mi.Scene'

    # FMCW configuration
    config: 'FMCWConfig'

    # Antenna arrays
    tx_array: 'TxArray'
    rx_array: 'RxArray'

    # Differentiable geometry (optional)
    vertex_materials: Optional['VertexMaterialParams'] = None
    vertex_positions: Optional['VertexPositions'] = None
    vertex_normals: Optional['VertexNormals'] = None

    # Edge/face data for diffraction (optional)
    edge_data: Optional['EdgeDataDr'] = None
    face_data: Optional['FaceData'] = None

    # Antenna patterns (optional)
    tx_pattern: Optional['AntennaPatternLoader'] = None
    rx_pattern: Optional['AntennaPatternLoader'] = None

    # Cached values
    epsilon: float = 1e-4

    @property
    def n_tx(self) -> int:
        """Number of TX elements."""
        return self.tx_array.num_elements

    @property
    def n_rx(self) -> int:
        """Number of RX elements."""
        return self.rx_array.num_elements

    @property
    def wavelength(self) -> float:
        """Wavelength at center frequency."""
        return self.config.c / self.config.center_freq

    @classmethod
    def from_files(cls,
                   config_file: str,
                   scene_file: str,
                   pattern_file: Optional[str] = None,
                   tx_pattern_file: Optional[str] = None,
                   rx_pattern_file: Optional[str] = None,
                   edge_file: Optional[str] = None,
                   material_type: str = "metal",
                   enable_gradients: bool = True,
                   verbose: bool = True) -> 'SceneContext':
        """
        Load SceneContext from files.

        Args:
            config_file: Path to JSON config file with TX/RX positions
            scene_file: Path to PLY mesh file
            pattern_file: Path to antenna pattern .npy file (used for both TX/RX)
            tx_pattern_file: Path to TX antenna pattern (overrides pattern_file)
            rx_pattern_file: Path to RX antenna pattern (overrides pattern_file)
            edge_file: Path to pre-computed edge data (optional)
            material_type: "metal", "dielectric", or "random"
            enable_gradients: Enable gradients on differentiable parameters
            verbose: Print loading progress

        Returns:
            Initialized SceneContext
        """
        from ..sensor.config import FMCWConfig, TxArray, RxArray
        from ..sensor.element_patterns import AntennaPatternLoader
        from .scene_params.vertex_positions import VertexPositions
        from .scene_params.vertex_normals import VertexNormals
        from .scene_params.vertex_materials import VertexMaterialParams

        if verbose:
            print("=" * 80)
            print("[SceneContext] Loading from files")
            print("=" * 80)

        # Step 1: Load FMCW configuration
        if verbose:
            print(f"\n[Step 1/6] Loading FMCW configuration from {config_file}")
        config = FMCWConfig.from_json(config_file)
        if verbose:
            print(f"  Center frequency: {config.center_freq/1e9:.2f} GHz")
            print(f"  Bandwidth: {config.bandwidth/1e9:.2f} GHz")
            print(f"  ADC samples: {config.num_adc_samples}")

        # Step 2: Load antenna patterns
        if verbose:
            print(f"\n[Step 2/6] Loading antenna patterns")

        if tx_pattern_file is None and rx_pattern_file is None:
            if pattern_file is None:
                raise ValueError(
                    "Must specify either pattern_file OR both tx_pattern_file and rx_pattern_file"
                )
            tx_pattern_path = pattern_file
            rx_pattern_path = pattern_file
            if verbose:
                print(f"  Using same pattern for TX and RX: {pattern_file}")
        else:
            if tx_pattern_file is None or rx_pattern_file is None:
                raise ValueError(
                    "If using separate patterns, must specify both tx_pattern_file and rx_pattern_file"
                )
            tx_pattern_path = tx_pattern_file
            rx_pattern_path = rx_pattern_file
            if verbose:
                print(f"  TX pattern: {tx_pattern_file}")
                print(f"  RX pattern: {rx_pattern_file}")

        tx_pattern = AntennaPatternLoader(tx_pattern_path)
        rx_pattern = AntennaPatternLoader(rx_pattern_path)

        # Step 3: Create TX/RX arrays
        if verbose:
            print(f"\n[Step 3/6] Creating TX/RX arrays")

        with open(config_file, 'r') as f:
            config_data = json.load(f)

        tx_array = TxArray.from_config(config_data, default_power=1.0)
        rx_array = RxArray.from_config(config_data)

        # Store pattern data in arrays
        tx_array.pattern_data = tx_pattern.pattern_data
        rx_array.pattern_data = rx_pattern.pattern_data
        tx_array.pattern_loader = tx_pattern
        rx_array.pattern_loader = rx_pattern

        if verbose:
            print(f"  TX array: {tx_array.num_elements} elements")
            print(f"  RX array: {rx_array.num_elements} elements")

        # Enable gradients for antenna parameters
        if enable_gradients:
            dr.enable_grad(tx_array.positions.x)
            dr.enable_grad(tx_array.positions.y)
            dr.enable_grad(tx_array.positions.z)
            dr.enable_grad(tx_array.orientations.x)
            dr.enable_grad(tx_array.orientations.y)
            dr.enable_grad(tx_array.orientations.z)
            dr.enable_grad(rx_array.positions.x)
            dr.enable_grad(rx_array.positions.y)
            dr.enable_grad(rx_array.positions.z)
            dr.enable_grad(rx_array.orientations.x)
            dr.enable_grad(rx_array.orientations.y)
            dr.enable_grad(rx_array.orientations.z)
            if verbose:
                print(f"  Gradients enabled for TX/RX positions and orientations")

            if tx_pattern is not None:
                tx_pattern.enable_gradients()
            if rx_pattern is not None:
                rx_pattern.enable_gradients()
            if verbose:
                print(f"  Gradients enabled for antenna patterns")

        # Step 4: Load scene geometry
        if verbose:
            print(f"\n[Step 4/6] Loading scene from {scene_file}")

        vertex_positions = VertexPositions.from_ply(scene_file, enable_grad=enable_gradients)

        # Step 5: Load vertex normals
        if verbose:
            print(f"\n[Step 5/6] Loading vertex normals")
        vertex_normals = VertexNormals.from_ply(scene_file, enable_grad=enable_gradients)

        # Step 6: Create vertex materials
        if verbose:
            print(f"\n[Step 6/6] Creating vertex materials ({material_type})")

        n_verts = vertex_positions.num_vertices
        if material_type == "metal":
            vertex_materials = VertexMaterialParams(
                num_vertices=n_verts,
                alpha=dr.full(mi.Float, 0.1, n_verts),
                eta=dr.full(mi.Float, 12.0, n_verts),
                kappa=dr.full(mi.Float, 85.0, n_verts),
                diffuse_albedo=dr.full(mi.Float, 0.0, n_verts),
            )
        elif material_type == "dielectric":
            vertex_materials = VertexMaterialParams(
                num_vertices=n_verts,
                alpha=dr.full(mi.Float, 0.05, n_verts),
                eta=dr.full(mi.Float, 2.5, n_verts),
                kappa=dr.full(mi.Float, 0.01, n_verts),
                diffuse_albedo=dr.full(mi.Float, 0.1, n_verts),
            )
        elif material_type == "random":
            import numpy as np
            rng = np.random.default_rng(42)
            vertex_materials = VertexMaterialParams(
                num_vertices=n_verts,
                alpha=mi.Float(rng.uniform(0.05, 0.3, n_verts)),
                eta=mi.Float(rng.uniform(2.0, 15.0, n_verts)),
                kappa=mi.Float(rng.uniform(0.1, 100.0, n_verts)),
                diffuse_albedo=mi.Float(rng.uniform(0.0, 0.2, n_verts)),
            )
        else:
            raise ValueError(f"Unknown material_type: {material_type}")

        if enable_gradients:
            vertex_materials.enable_gradients()

        # Build Mitsuba scene from PLY file
        scene = cls._build_mitsuba_scene(scene_file)

        # Compute scene epsilon
        bbox = scene.bbox()
        diagonal = float(dr.norm(bbox.max - bbox.min))
        epsilon = diagonal * 1e-5
        epsilon = max(1e-6, min(epsilon, 1e-2))

        if verbose:
            print(f"\n[SceneContext] Initialization complete")
            print(f"  Scene bounding box diagonal: {diagonal:.2f} m")
            print(f"  Shadow ray epsilon: {epsilon:.2e} m")

        return cls(
            scene=scene,
            config=config,
            tx_array=tx_array,
            rx_array=rx_array,
            vertex_materials=vertex_materials,
            vertex_positions=vertex_positions,
            vertex_normals=vertex_normals,
            edge_data=None,
            face_data=None,
            tx_pattern=tx_pattern,
            rx_pattern=rx_pattern,
            epsilon=epsilon
        )

    @staticmethod
    def _build_mitsuba_scene(scene_file: str) -> 'mi.Scene':
        """Build Mitsuba scene from PLY file."""
        scene_dict = {
            'type': 'scene',
            'mesh': {
                'type': 'ply',
                'filename': scene_file,
                'bsdf': {
                    'type': 'conductor',
                    'material': 'Al'
                }
            }
        }
        scene = mi.load_dict(scene_dict)
        return scene

    def load_edge_data(self, edge_file: Optional[str] = None,
                       extract_from_mesh: bool = True,
                       verbose: bool = True) -> None:
        """Load or extract edge data for diffraction."""
        from .scene_params.edge_data import EdgeDataDr
        from .scene_params.extract_edges_from_mesh import extract_edges_from_mesh

        if edge_file is not None:
            if verbose:
                print(f"[SceneContext] Loading edge data from {edge_file}")
            self.edge_data = EdgeDataDr.load(edge_file)
        elif extract_from_mesh and self.vertex_positions is not None:
            if verbose:
                print(f"[SceneContext] Extracting edges from mesh")
            vertices = self.vertex_positions.get_vertices_numpy()
            faces = self.vertex_positions.get_faces_numpy()
            self.edge_data = extract_edges_from_mesh(vertices, faces)

        if verbose and self.edge_data is not None:
            print(f"  Loaded {self.edge_data.num_edges} edges")


class RenderLogger:
    """
    Hierarchical logging with toggleable verbosity.

    Verbosity levels:
    - 0: Silent (no output)
    - 1: Summary only (start/end, final stats)
    - 2: Per-bounce information
    - 3: Debug details (MIS weights, E-field stats)
    """

    def __init__(self, verbosity: int = 0):
        self.verbosity = verbosity
        self._indent = 0

    def _print(self, msg: str, min_verbosity: int = 1) -> None:
        """Print message if verbosity is sufficient."""
        if self.verbosity >= min_verbosity:
            indent = "  " * self._indent
            print(f"{indent}{msg}")

    def log_render_start(self, n_samples: int, max_depth: int, seed: int) -> None:
        self._print(f"[RENDER] Starting: {n_samples:,} samples, depth={max_depth}, seed={seed}", 1)

    def log_render_end(self, total_paths: int, elapsed_ms: float) -> None:
        self._print(f"[RENDER] Complete: {total_paths:,} paths, {elapsed_ms:.1f} ms", 1)

    def log_bounce_start(self, depth: int, n_active: int) -> None:
        self._print(f"[Bounce {depth}] Active paths: {n_active:,}", 2)
        self._indent += 1

    def log_bounce_end(self, depth: int, stats: Dict[str, int]) -> None:
        self._indent -= 1
        self._print(f"[Bounce {depth}] NEE: {stats.get('nee', 0):,}, Cont: {stats.get('continue', 0):,}", 2)

    def log_nee(self, n_valid: int, n_total: int) -> None:
        self._print(f"NEE: {n_valid:,}/{n_total:,} valid", 2)

    def log_mis_weights(self, w_nee: float, w_edge: float, w_cont: float) -> None:
        self._print(f"MIS: w_nee={w_nee:.4f}, w_edge={w_edge:.4f}, w_cont={w_cont:.4f}", 3)

    def log_efield_stats(self, mean_mag: float, max_mag: float) -> None:
        self._print(f"E-field: mean={mean_mag:.4e}, max={max_mag:.4e}", 3)
