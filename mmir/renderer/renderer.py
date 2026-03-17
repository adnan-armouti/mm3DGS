"""
RRTS Reference Renderer - Main orchestrator class.

This module provides FMCWRendererRef, the main renderer class that coordinates
reservoir sampling and ADC synthesis to match RRTS output exactly.

Usage:
    renderer = FMCWRendererRef.from_files(
        mesh_file="scene.ply",
        config_file="config.json",
        pattern_file="patterns.npz"
    )
    result = renderer.render(seed=42)
    adc = renderer.get_adc_numpy()
"""

from typing import Optional, TYPE_CHECKING
import drjit as dr
import mitsuba as mi
import numpy as np

# Scene context and logging (inlined)
from .scene_context import SceneContext, RenderLogger

from .config import RenderConfigRef, RenderResultRef
from .sampler import ReservoirSampler, ReservoirHits
from .integrator import SBRIntegratorRef, ADCResult
from .specular.hashing import SpecularDeduplicator
from .specular.image_method import ImageMethodRefiner, SpecularPaths
from .specular.sms import SpecularManifoldSampler
from .materials.parameterization import MaterialParameterization

# Import diffraction modules (optional)
try:
    from .diffraction import DiffractionConfig, FsdBSDF
    from .diffraction.triangle_search import build_triangle_hash_from_scene
    _HAS_DIFFRACTION = True
except ImportError:
    _HAS_DIFFRACTION = False

# Import RRTS-style pattern loading and GPU-native pattern evaluation
try:
    from mmir.sensor.element_patterns import load_pattern_rrts_format, AntennaPatternLoader
except ImportError:
    load_pattern_rrts_format = None
    AntennaPatternLoader = None

# Import pattern importance sampler (Phase 1)
try:
    from .utils.pattern_sampler import PatternImportanceSampler
except ImportError:
    PatternImportanceSampler = None

if TYPE_CHECKING:
    from mmir.sensor.config import FMCWConfig


class FMCWRendererRef:
    """
    RRTS Reference Renderer.

    This renderer matches RRTS output by using:
    - RX-centric reservoir sampling
    - RRTS-exact BRDF (Diffuse_light)
    - RRTS-exact time grid (t[k] = k/sample_rate)
    - RRTS-exact phase computation

    The interface is designed to be a drop-in replacement for FMCWRenderer
    for testing and validation purposes.
    """

    def __init__(
        self,
        scene_ctx: SceneContext,
        config: Optional[RenderConfigRef] = None,
        verbose: bool = True,
        tx_pattern_file: Optional[str] = None,
        rx_pattern_file: Optional[str] = None
    ):
        """
        Initialize RRTS reference renderer.

        Args:
            scene_ctx: Scene context with loaded scene, config, patterns
            config: Render configuration (uses defaults if None)
            verbose: Print progress information
            tx_pattern_file: Optional path to TX antenna pattern for RRTS-style loading
            rx_pattern_file: Optional path to RX antenna pattern for RRTS-style loading
        """
        self.scene_ctx = scene_ctx
        self.config = config or RenderConfigRef()
        self.verbose = verbose
        self.logger = RenderLogger(verbose)

        # Store pattern file paths for RRTS-style loading
        self.tx_pattern_file = tx_pattern_file
        self.rx_pattern_file = rx_pattern_file

        # Optional per-triangle materials for iterative optimization
        # Shape: [n_triangles, 3] or [n_triangles, 6]
        # 3 columns: (albedo, roughness, metallic) - legacy mode
        # 6 columns: (eps_real, eps_imag, sigma_h, l_c, tau, thickness) - physics mode
        # If None, sampler uses default materials
        self.triangle_materials: Optional[np.ndarray] = None
        # GPU-resident copy: list of mi.Float, one per material column
        self._triangle_materials_gpu: Optional[list] = None
        self._triangle_materials_gpu_src = None  # identity tracking for cache invalidation

        # Load RRTS-style patterns if available
        self.tx_pattern_rrts = None
        self.rx_pattern_rrts = None
        if load_pattern_rrts_format is not None:
            if tx_pattern_file is not None:
                try:
                    self.tx_pattern_rrts = load_pattern_rrts_format(tx_pattern_file)
                except Exception as e:
                    if verbose:
                        print(f"[Warning] Could not load TX RRTS pattern: {e}")
            if rx_pattern_file is not None:
                try:
                    self.rx_pattern_rrts = load_pattern_rrts_format(rx_pattern_file)
                except Exception as e:
                    if verbose:
                        print(f"[Warning] Could not load RX RRTS pattern: {e}")

        # Load GPU-native pattern loaders (for fast antenna evaluation in DrJit)
        self.tx_pattern_loader = None
        self.rx_pattern_loader = None
        if AntennaPatternLoader is not None:
            if tx_pattern_file is not None:
                try:
                    self.tx_pattern_loader = AntennaPatternLoader(tx_pattern_file)
                except Exception as e:
                    if verbose:
                        print(f"[Warning] Could not create TX AntennaPatternLoader: {e}")
            if rx_pattern_file is not None:
                try:
                    self.rx_pattern_loader = AntennaPatternLoader(rx_pattern_file)
                except Exception as e:
                    if verbose:
                        print(f"[Warning] Could not create RX AntennaPatternLoader: {e}")

        # Initialize pattern importance sampler if enabled (Phase 1)
        self.pattern_sampler = None
        if self.config.use_pattern_importance_sampling and self.rx_pattern_rrts is not None:
            if PatternImportanceSampler is not None:
                try:
                    self.pattern_sampler = PatternImportanceSampler(
                        pattern_data=self.rx_pattern_rrts,
                        n_theta=self.config.pattern_sampler_n_theta,
                        n_phi=self.config.pattern_sampler_n_phi,
                        combine_mode='product'
                    )
                    if verbose:
                        print(f"[PatternImportanceSampler] Created for RX pattern")
                        print(f"  Grid: {self.config.pattern_sampler_n_theta} theta × {self.config.pattern_sampler_n_phi} phi bins")
                except Exception as e:
                    if verbose:
                        print(f"[Warning] Could not create pattern importance sampler: {e}")
            else:
                if verbose:
                    print("[Warning] PatternImportanceSampler not available, using cosine sampling")

        # Initialize components
        self.sampler = ReservoirSampler(
            n_hits_per_rx=self.config.n_hits_per_rx,
            n_rays_per_res=self.config.n_rays_per_res,
            max_distance=self.config.max_distance,
            pattern_sampler=self.pattern_sampler,
            hemisphere_sampling=getattr(self.config, 'hemisphere_sampling', 'cosine'),
        )

        self.integrator = SBRIntegratorRef(scene_ctx.config, self.config)

        # Material parameterization strategy (default: None = per-triangle fallback)
        self._material_param: Optional[MaterialParameterization] = None

        # Initialize SMS solver (if enabled)
        self.sms_solver = None
        if self.config.enable_sms:
            self.sms_solver = SpecularManifoldSampler(
                wavelength=self.integrator.wavelength,
                bsdf=self.integrator.bsdf,
                max_iterations=self.config.sms_max_iterations,
                solver_threshold=self.config.sms_solver_threshold,
                use_smooth_normals=self.config.sms_use_smooth_normals,
            )
            if verbose:
                print(f"[SMS] Specular Manifold Sampling enabled "
                      f"(max_iter={self.config.sms_max_iterations}, "
                      f"thresh={self.config.sms_solver_threshold:.1e})")

        # Initialize image method components (if enabled)
        self.deduplicator = None
        self.image_refiner = None
        self.patch_data = None
        if self.config.enable_image_method or self.config.enable_sms:
            self.deduplicator = SpecularDeduplicator(
                counter_size=self.config.specular_dedup_counter_size,
                num_hash_functions=self.config.num_hash_functions,
            )
            self.image_refiner = ImageMethodRefiner(
                wavelength=self.integrator.wavelength,
                bsdf=self.integrator.bsdf,
            )
            if verbose:
                print(f"[ImageMethod] Enabled with counter_size={self.config.specular_dedup_counter_size}, "
                      f"num_hashes={self.config.num_hash_functions}")

            # Initialize patch clustering (if enabled)
            if self.config.use_patch_clustering:
                from .utils.clustering import PatchClusterer
                clusterer = PatchClusterer(
                    angle_threshold_deg=self.config.patch_angle_threshold_deg,
                    distance_threshold=self.config.patch_distance_threshold,
                    max_tris_per_patch=self.config.patch_max_tris,
                    use_spatial_adjacency=self.config.use_spatial_adjacency,
                    spatial_radius=self.config.spatial_radius,
                    verbose=verbose,
                )
                self.patch_data = clusterer.build_patches_from_scene(scene_ctx.scene)
                if verbose:
                    print(f"[PatchClustering] {self.patch_data.n_patches} patches, "
                          f"max_tris_per_patch={self.patch_data.max_tris_per_patch}")

        # Initialize free-space diffraction (if enabled)
        if (_HAS_DIFFRACTION and self.config.diffraction_config is not None
                and self.config.diffraction_config.enabled):
            diff_cfg = self.config.diffraction_config
            wavelength = self.integrator.wavelength
            k = 2.0 * np.pi / wavelength
            beam_sigma = diff_cfg.beam_sigma_wavelengths * wavelength

            # Auto-compute hash cell size if not specified
            cell_size = diff_cfg.hash_cell_size
            if cell_size <= 0:
                cell_size = 3.0 * beam_sigma  # search radius

            # Build spatial hash from scene mesh
            tri_hash = build_triangle_hash_from_scene(
                scene_ctx.scene, cell_size,
                edge_angle_threshold_deg=diff_cfg.edge_angle_threshold_deg,
                wavelength=wavelength,
                min_edge_length_wavelengths=diff_cfg.min_edge_length_wavelengths,
                boundary_erosion_hops=diff_cfg.boundary_erosion_hops,
                region_angle_threshold_deg=diff_cfg.region_angle_threshold_deg,
                region_dihedral_cap_deg=diff_cfg.region_dihedral_cap_deg,
                min_region_faces=diff_cfg.min_region_faces,
                inter_region_min_faces=diff_cfg.inter_region_min_faces,
                min_chain_length_wavelengths=diff_cfg.min_chain_length_wavelengths,
                chain_collinearity_threshold_deg=diff_cfg.chain_collinearity_threshold_deg,
            )

            # Create FsdBSDF evaluator
            fsd_bsdf = FsdBSDF(k=k, beam_sigma=beam_sigma, beta_max=diff_cfg.beta_max)

            # Wire into the integrator
            self.integrator.tri_hash = tri_hash
            self.integrator.fsd_bsdf = fsd_bsdf
            self.integrator.diffraction_config = diff_cfg

            # Build GPU spatial hash (always used; CPU path is deprecated)
            from .diffraction.fsd_aperture_gpu import build_gpu_spatial_hash
            self.integrator._gpu_spatial_hash = build_gpu_spatial_hash(tri_hash)
            if verbose:
                print(f"  [FSD] GPU aperture construction enabled")

            # Initialize FSD CDF tables for importance sampling (Tier 2)
            direction_sampling = getattr(self.config, 'multibounce_direction_sampling', 'cosine')
            fsd_sampling_enabled = getattr(self.config, 'fsd_sampling_enabled', True)
            if direction_sampling == 'bsdf' and fsd_sampling_enabled:
                from .diffraction.fsd_sampling_tables import get_fsd_tables
                cdf_resolution = getattr(self.config, 'fsd_cdf_resolution', 1024)
                fsd_tables = get_fsd_tables()
                fsd_tables.upload_to_gpu()
                self.integrator._fsd_cdf_tables = fsd_tables
                self.integrator._fsd_sir_candidates = getattr(
                    self.config, 'fsd_sir_candidates', 8)
                if verbose:
                    print(f"  [FSD] CDF tables loaded ({cdf_resolution}x{cdf_resolution}), "
                          f"SIR candidates={self.integrator._fsd_sir_candidates}")

            if verbose:
                search_r = 3.0 * beam_sigma
                print(f"[Diffraction] fsdBSDF enabled")
                print(f"  λ={wavelength*1e3:.2f}mm, k={k:.1f} m⁻¹, "
                      f"σ={beam_sigma*1e3:.1f}mm, search_r={search_r*1e3:.1f}mm")
                print(f"  energy_borrowing={diff_cfg.energy_borrowing}, β_max={diff_cfg.beta_max}")

        # Initialize boundary gradient computer (if enabled)
        self.boundary_computer = None
        if self.config.enable_boundary_gradients:
            from .utils.boundary import BoundaryGradientComputer
            self.boundary_computer = BoundaryGradientComputer(
                config=self.config,
                integrator=self.integrator,
            )
            if verbose:
                print(f"[Boundary] Gradient computer initialized "
                      f"(n_samples={self.config.n_boundary_samples_primary})")

        # Store last render result
        self._last_result: Optional[RenderResultRef] = None

    def _update_diffraction_materials(self):
        """Propagate per-triangle eps_real/eps_imag to integrator for material-dependent diffraction."""
        diff_cfg = self.config.diffraction_config
        if (diff_cfg is not None and diff_cfg.enabled and diff_cfg.material_opacity
                and self.triangle_materials is not None
                and self.triangle_materials.shape[1] >= 2):
            self.integrator.diffraction_tri_eps = (
                self.triangle_materials[:, 0].astype(np.float32),  # eps_real
                self.triangle_materials[:, 1].astype(np.float32),  # eps_imag
            )
        else:
            self.integrator.diffraction_tri_eps = (None, None)

    @property
    def material_param(self) -> Optional[MaterialParameterization]:
        """Material parameterization strategy."""
        return self._material_param

    @material_param.setter
    def material_param(self, value: Optional[MaterialParameterization]):
        """Set material parameterization and propagate to integrator."""
        self._material_param = value
        self.integrator.material_param = value

    @classmethod
    def from_files(
        cls,
        mesh_file: str,
        config_file: str,
        pattern_file: Optional[str] = None,
        tx_pattern_file: Optional[str] = None,
        rx_pattern_file: Optional[str] = None,
        material_type: str = "metal",
        render_config: Optional[RenderConfigRef] = None,
        verbose: bool = True
    ) -> 'FMCWRendererRef':
        """
        Create renderer from files.

        Factory method that loads scene, config, and patterns, then creates
        the renderer. This provides the same interface as FMCWRenderer.from_files().

        Args:
            mesh_file: Path to mesh file (.ply)
            config_file: Path to FMCW config file (.json)
            pattern_file: Optional path to antenna patterns (.npy) for both TX/RX
            tx_pattern_file: Optional path to TX antenna pattern (overrides pattern_file)
            rx_pattern_file: Optional path to RX antenna pattern (overrides pattern_file)
            material_type: Material type ("metal", "dielectric", "random")
            render_config: Optional render configuration
            verbose: Print progress

        Returns:
            Configured FMCWRendererRef instance
        """
        # Use SceneContext.from_files() to load everything
        # SceneContext.from_files() expects: config_file, scene_file, pattern_file, etc.
        scene_ctx = SceneContext.from_files(
            config_file=config_file,
            scene_file=mesh_file,  # SceneContext calls it scene_file
            pattern_file=pattern_file,
            tx_pattern_file=tx_pattern_file,
            rx_pattern_file=rx_pattern_file,
            material_type=material_type,
            enable_gradients=False,  # renderer_ref is for forward pass only
            verbose=verbose
        )

        # Determine pattern files for RRTS-style loading
        actual_tx_pattern = tx_pattern_file or pattern_file
        actual_rx_pattern = rx_pattern_file or pattern_file

        return cls(
            scene_ctx,
            render_config,
            verbose,
            tx_pattern_file=actual_tx_pattern,
            rx_pattern_file=actual_rx_pattern
        )

    def render(self, seed: Optional[int] = None) -> RenderResultRef:
        """
        Execute RRTS-style rendering.

        Steps:
        1. Reservoir sampling from each RX element
        2. ADC synthesis connecting hits to all TX elements

        Args:
            seed: Random seed (uses config seed if None)

        Returns:
            RenderResultRef containing ADC and reservoir data
        """
        seed = seed or self.config.seed

        # Release previous result to free DrJit arrays
        if self._last_result is not None:
            self._last_result = None

        if self.verbose:
            print("\n" + "=" * 60)
            print("RRTS Reference Renderer")
            print("=" * 60)
            print(f"Seed: {seed}")
            print(f"n_hits_per_rx: {self.config.n_hits_per_rx}")
            print(f"n_rays_per_res: {self.config.n_rays_per_res}")

        # Get RX positions and boresights
        rx_array = self.scene_ctx.rx_array
        n_rx = rx_array.num_elements

        rx_positions = mi.Point3f(
            mi.Float(rx_array.positions[:, 0]),
            mi.Float(rx_array.positions[:, 1]),
            mi.Float(rx_array.positions[:, 2])
        )

        # Use orientations (boresight directions) from RxArray
        # RxArray.orientations is a mi.Vector3f with per-element boresight directions
        if hasattr(rx_array, 'orientations') and rx_array.orientations is not None:
            rx_boresights = rx_array.orientations
        else:
            # Default boresight: +y (forward direction in our coordinate system)
            rx_boresights = mi.Vector3f(
                mi.Float(np.zeros(n_rx)),
                mi.Float(np.ones(n_rx)),  # +Y is forward/boresight
                mi.Float(np.zeros(n_rx))
            )

        # Propagate material arrays for material-dependent diffraction
        self._update_diffraction_materials()

        # Step 1: Reservoir sampling
        if self.verbose:
            print("\n[Step 1] Reservoir Sampling")

        reservoir_hits = self.sampler.sample_vectorized(
            scene=self.scene_ctx.scene,
            rx_positions=rx_positions,
            rx_boresights=rx_boresights,
            seed=seed,
            verbose=self.verbose,
            triangle_materials=self.triangle_materials,
            use_vertex_normals=self.config.use_vertex_normals,
        )

        # Get TX positions and boresights
        tx_array = self.scene_ctx.tx_array
        n_tx = tx_array.num_elements

        tx_positions = mi.Point3f(
            mi.Float(tx_array.positions[:, 0]),
            mi.Float(tx_array.positions[:, 1]),
            mi.Float(tx_array.positions[:, 2])
        )

        # Use orientations (boresight directions) from TxArray
        if hasattr(tx_array, 'orientations') and tx_array.orientations is not None:
            tx_boresights = tx_array.orientations
        else:
            # Default boresight: +y (forward direction in our coordinate system)
            tx_boresights = mi.Vector3f(
                mi.Float(np.zeros(n_tx)),
                mi.Float(np.ones(n_tx)),  # +Y is forward/boresight
                mi.Float(np.zeros(n_tx))
            )

        # Step 2: Specular path pipeline (SMS or image method, if enabled)
        specular_paths = None
        use_non_ka = False
        patch_has_specular = None
        tri_to_patch = None

        _specular_enabled = (self.config.enable_sms or self.config.enable_image_method) and self.deduplicator is not None
        if _specular_enabled:
            use_patches = self.patch_data is not None

            if self.verbose:
                method_name = "SMS" if self.config.enable_sms else "Image Method"
                mode_str = "PATCH" if use_patches else "TRIANGLE"
                print(f"\n[Step 2] {method_name} Pipeline ({mode_str} mode)")

            # Step 2a: Deduplicate specular paths
            if self.verbose:
                print("  [Step 2a] Specular path deduplication")

            unique_mask, dedup_stats = self.deduplicator.deduplicate(
                hit_N=reservoir_hits.hit_N,
                hit_P=reservoir_hits.hit_P,
                valid=reservoir_hits.valid,
                n_rx=n_rx,
                use_patches=use_patches,
                hit_ID=reservoir_hits.hit_ID if use_patches else None,
                patch_data=self.patch_data,
            )

            dedup_label = "patches" if dedup_stats.get('use_patches') else "triangles"
            if self.verbose:
                print(f"    Valid hits: {dedup_stats['n_valid_hits']:,}")
                print(f"    Unique {dedup_label}: {dedup_stats['n_unique_triangles']:,}")
                print(f"    Duplicates removed: {dedup_stats['n_duplicates_removed']:,}")
                print(f"    Dedup ratio: {dedup_stats['dedup_ratio']:.3f}")

            # Step 2b: Specular path finding (SMS or image method)
            if self.config.enable_sms and self.sms_solver is not None:
                # ---- SMS path ----
                if self.verbose:
                    print("  [Step 2b] SMS Newton solver")

                specular_paths = self.sms_solver.find_specular_paths(
                    unique_mask=unique_mask,
                    hit_P=reservoir_hits.hit_P,
                    hit_N=reservoir_hits.hit_N,
                    hit_rho=reservoir_hits.hit_rho,
                    hit_ID=reservoir_hits.hit_ID,
                    tx_positions=tx_positions,
                    rx_positions=rx_positions,
                    n_hits_per_rx=self.config.n_hits_per_rx,
                    scene=self.scene_ctx.scene,
                    verbose=self.verbose,
                    patch_data=self.patch_data,
                    triangle_materials=self.triangle_materials,
                )
                n_valid_spec = int(dr.sum(mi.UInt32(specular_paths.valid))[0]) if specular_paths.n_paths > 0 else 0
                n_total_hits = dedup_stats['n_valid_hits']
                n_unique = dedup_stats['n_unique_triangles']
                n_deduped = dedup_stats['n_duplicates_removed']
                print(f"  [SMS] Diffuse hits: {n_total_hits:,} | "
                      f"Unique {dedup_label}: {n_unique:,} (dedup removed {n_deduped:,}) | "
                      f"Valid specular paths: {n_valid_spec:,}/{specular_paths.n_paths:,}")

                # Per-patch adaptive BSDF: build mask of which patches have specular coverage
                if use_patches and self.patch_data is not None and specular_paths.patch_id is not None and n_valid_spec > 0:
                    n_patches = self.patch_data.n_patches
                    patch_has_specular_np = np.zeros(n_patches, dtype=bool)
                    valid_sp_np = np.array(specular_paths.valid)
                    patch_id_np = np.array(specular_paths.patch_id)
                    valid_patch_ids = patch_id_np[valid_sp_np].astype(np.int64)
                    valid_patch_ids = valid_patch_ids[(valid_patch_ids >= 0) & (valid_patch_ids < n_patches)]
                    patch_has_specular_np[valid_patch_ids] = True
                    n_covered = int(np.sum(patch_has_specular_np))
                    print(f"  [Per-Patch BSDF] {n_covered}/{n_patches} patches have specular coverage")

                    patch_has_specular = mi.Bool(patch_has_specular_np)
                    tri_to_patch = self.patch_data.tri_to_patch
                    use_non_ka = True  # Enable per-hit selection in integrator
                else:
                    if n_valid_spec > 0:
                        use_non_ka = True
                    else:
                        use_non_ka = False
                        print(f"  [SMS] 0 valid specular paths → using full BSDF (KA retained in MC)")

            elif reservoir_hits.hit_tri_v0 is not None:
                # ---- Image method path ----
                if self.verbose:
                    print("  [Step 2b] Image method refinement")

                specular_paths = self.image_refiner.refine(
                    unique_mask=unique_mask,
                    hit_P=reservoir_hits.hit_P,
                    hit_N=reservoir_hits.hit_N,
                    hit_rho=reservoir_hits.hit_rho,
                    hit_tri_v0=reservoir_hits.hit_tri_v0,
                    hit_tri_v1=reservoir_hits.hit_tri_v1,
                    hit_tri_v2=reservoir_hits.hit_tri_v2,
                    tx_positions=tx_positions,
                    rx_positions=rx_positions,
                    n_hits_per_rx=self.config.n_hits_per_rx,
                    scene=self.scene_ctx.scene,
                    verbose=self.verbose,
                    use_patches=use_patches,
                    hit_ID=reservoir_hits.hit_ID,
                    patch_data=self.patch_data,
                )
                # Always print image method stats (user-requested)
                n_valid_spec = int(dr.sum(mi.UInt32(specular_paths.valid))[0]) if specular_paths.n_paths > 0 else 0
                n_total_hits = dedup_stats['n_valid_hits']
                n_unique = dedup_stats['n_unique_triangles']
                n_deduped = dedup_stats['n_duplicates_removed']
                print(f"  [ImageMethod] Diffuse hits: {n_total_hits:,} | "
                      f"Unique {dedup_label}: {n_unique:,} (dedup removed {n_deduped:,}) | "
                      f"Valid specular paths: {n_valid_spec:,}/{specular_paths.n_paths:,}")

                # Per-patch adaptive BSDF: build mask of which patches have specular coverage
                if use_patches and self.patch_data is not None and specular_paths.patch_id is not None:
                    n_patches = self.patch_data.n_patches
                    patch_has_specular_np = np.zeros(n_patches, dtype=bool)
                    valid_sp_np = np.array(specular_paths.valid)
                    patch_id_np = np.array(specular_paths.patch_id)
                    valid_patch_ids = patch_id_np[valid_sp_np].astype(np.int64)
                    valid_patch_ids = valid_patch_ids[(valid_patch_ids >= 0) & (valid_patch_ids < n_patches)]
                    patch_has_specular_np[valid_patch_ids] = True
                    n_covered = int(np.sum(patch_has_specular_np))
                    print(f"  [Per-Patch BSDF] {n_covered}/{n_patches} patches have specular coverage")

                    patch_has_specular = mi.Bool(patch_has_specular_np)
                    tri_to_patch = self.patch_data.tri_to_patch
                    use_non_ka = True  # Enable image method branch (per-hit selection in integrator)
                else:
                    # No patch data: fall back to scene-level adaptive BSDF
                    if n_valid_spec > 0:
                        use_non_ka = True
                    else:
                        use_non_ka = False
                        print(f"  [Adaptive BSDF] 0 valid specular paths → using full BSDF (KA retained in MC)")
            else:
                print("  [ImageMethod] WARNING: Triangle vertex data not available, skipping")

        # Step 3: ADC synthesis
        if self.verbose:
            synthesis_mode = "(per-patch dual-path)" if patch_has_specular is not None else \
                             "(dual-path)" if use_non_ka else "(standard)"
            print(f"\n[Step 3] ADC Synthesis {synthesis_mode}")

        # Sync triangle materials to integrator for Phase A specular BSDF evaluation
        self.integrator.triangle_materials = self.triangle_materials

        if self.config.use_shared_hits:
            # RRTS-style: each hit processed for ALL (TX, RX) pairs
            # Fully vectorized ADC synthesis (~135-210x faster than loop version)
            adc_result = self.integrator.synthesize_shared_hits_vectorized(
                tx_positions=tx_positions,
                rx_positions=rx_positions,
                tx_boresights=tx_boresights,  # For antenna pattern evaluation
                rx_boresights=rx_boresights,  # For antenna pattern evaluation + sampling PDF
                reservoir_hits=reservoir_hits,
                scene=self.scene_ctx.scene,
                tx_pattern_rrts=self.tx_pattern_rrts,
                rx_pattern_rrts=self.rx_pattern_rrts,
                tx_pattern_loader=self.tx_pattern_loader,
                rx_pattern_loader=self.rx_pattern_loader,
                antenna_gain_linear=self.config.antenna_gain_linear,
                verbose=self.verbose,
                use_non_ka_bsdf=use_non_ka,
                specular_paths=specular_paths,
                patch_has_specular=patch_has_specular,
                tri_to_patch=tri_to_patch,
            )
        else:
            # RX-owned hits: each hit only processed with its owning RX
            # Fully vectorized (~160-210x faster than loop version)
            adc_result = self.integrator.synthesize_vectorized(
                tx_positions=tx_positions,
                rx_positions=rx_positions,
                reservoir_hits=reservoir_hits,
                scene=self.scene_ctx.scene,
                verbose=self.verbose
            )

        # Store result
        self._last_result = RenderResultRef(
            adc_result=adc_result,
            reservoir_hits=reservoir_hits,
            config=self.config
        )

        if self.verbose:
            print("\n" + "=" * 60)
            print("Rendering Complete")
            print(f"  Total paths: {adc_result.total_paths:,}")
            print(f"  ADC shape: ({adc_result.n_tx}, {adc_result.n_rx}, {adc_result.n_samples})")
            print("=" * 60)

        return self._last_result

    def render_with_cached_geometry(self, seed: Optional[int] = None):
        """
        Execute rendering and return both the result and cached geometry.

        This is Phase A of the two-phase differentiable render pipeline.
        The cached geometry contains all material-independent intermediates
        needed for Phase B (differentiable re-synthesis).

        Args:
            seed: Random seed (uses config seed if None)

        Returns:
            (RenderResultRef, CachedGeometry) tuple. CachedGeometry may be None
            if no valid hits or if hit_ID is unavailable.
        """
        seed = seed or self.config.seed

        # Release previous result
        if self._last_result is not None:
            self._last_result = None

        # --- Setup identical to render() ---
        rx_array = self.scene_ctx.rx_array
        n_rx = rx_array.num_elements
        rx_positions = mi.Point3f(
            mi.Float(rx_array.positions[:, 0]),
            mi.Float(rx_array.positions[:, 1]),
            mi.Float(rx_array.positions[:, 2])
        )
        if hasattr(rx_array, 'orientations') and rx_array.orientations is not None:
            rx_boresights = rx_array.orientations
        else:
            rx_boresights = mi.Vector3f(
                mi.Float(np.zeros(n_rx)), mi.Float(np.ones(n_rx)), mi.Float(np.zeros(n_rx))
            )

        # Propagate material arrays for material-dependent diffraction
        self._update_diffraction_materials()

        # Reservoir sampling
        reservoir_hits = self.sampler.sample_vectorized(
            scene=self.scene_ctx.scene,
            rx_positions=rx_positions, rx_boresights=rx_boresights,
            seed=seed, verbose=self.verbose,
            triangle_materials=self.triangle_materials,
            use_vertex_normals=self.config.use_vertex_normals,
        )

        tx_array = self.scene_ctx.tx_array
        n_tx = tx_array.num_elements
        tx_positions = mi.Point3f(
            mi.Float(tx_array.positions[:, 0]),
            mi.Float(tx_array.positions[:, 1]),
            mi.Float(tx_array.positions[:, 2])
        )
        if hasattr(tx_array, 'orientations') and tx_array.orientations is not None:
            tx_boresights = tx_array.orientations
        else:
            tx_boresights = mi.Vector3f(
                mi.Float(np.zeros(n_tx)), mi.Float(np.ones(n_tx)), mi.Float(np.zeros(n_tx))
            )

        # Specular path pipeline (SMS or image method, same logic as render())
        specular_paths = None
        use_non_ka = False
        patch_has_specular = None
        tri_to_patch = None

        _specular_enabled = (self.config.enable_sms or self.config.enable_image_method) and self.deduplicator is not None
        if _specular_enabled:
            use_patches = self.patch_data is not None

            unique_mask, dedup_stats = self.deduplicator.deduplicate(
                hit_N=reservoir_hits.hit_N,
                hit_P=reservoir_hits.hit_P,
                valid=reservoir_hits.valid,
                n_rx=n_rx,
                use_patches=use_patches,
                hit_ID=reservoir_hits.hit_ID if use_patches else None,
                patch_data=self.patch_data,
            )

            dedup_label = "patches" if dedup_stats.get('use_patches') else "triangles"

            if self.config.enable_sms and self.sms_solver is not None:
                # ---- SMS path ----
                specular_paths = self.sms_solver.find_specular_paths(
                    unique_mask=unique_mask,
                    hit_P=reservoir_hits.hit_P,
                    hit_N=reservoir_hits.hit_N,
                    hit_rho=reservoir_hits.hit_rho,
                    hit_ID=reservoir_hits.hit_ID,
                    tx_positions=tx_positions,
                    rx_positions=rx_positions,
                    n_hits_per_rx=self.config.n_hits_per_rx,
                    scene=self.scene_ctx.scene,
                    verbose=self.verbose,
                    patch_data=self.patch_data,
                    triangle_materials=self.triangle_materials,
                )
                n_valid_spec = int(dr.sum(mi.UInt32(specular_paths.valid))[0]) if specular_paths.n_paths > 0 else 0
                print(f"  [SMS] Unique {dedup_label}: {dedup_stats['n_unique_triangles']:,} | "
                      f"Valid specular paths: {n_valid_spec:,}/{specular_paths.n_paths:,}")

                # Per-patch adaptive BSDF: build mask of which patches have specular coverage
                if use_patches and self.patch_data is not None and specular_paths.patch_id is not None and n_valid_spec > 0:
                    n_patches = self.patch_data.n_patches
                    patch_has_specular_np = np.zeros(n_patches, dtype=bool)
                    valid_sp_np = np.array(specular_paths.valid)
                    patch_id_np = np.array(specular_paths.patch_id)
                    valid_patch_ids = patch_id_np[valid_sp_np].astype(np.int64)
                    valid_patch_ids = valid_patch_ids[(valid_patch_ids >= 0) & (valid_patch_ids < n_patches)]
                    patch_has_specular_np[valid_patch_ids] = True
                    n_covered = int(np.sum(patch_has_specular_np))
                    print(f"  [Per-Patch BSDF] {n_covered}/{n_patches} patches have specular coverage")

                    patch_has_specular = mi.Bool(patch_has_specular_np)
                    tri_to_patch = self.patch_data.tri_to_patch
                    use_non_ka = True
                else:
                    if n_valid_spec > 0:
                        use_non_ka = True
                    else:
                        use_non_ka = False

            elif reservoir_hits.hit_tri_v0 is not None:
                # ---- Image method path ----
                specular_paths = self.image_refiner.refine(
                    unique_mask=unique_mask,
                    hit_P=reservoir_hits.hit_P,
                    hit_N=reservoir_hits.hit_N,
                    hit_rho=reservoir_hits.hit_rho,
                    hit_tri_v0=reservoir_hits.hit_tri_v0,
                    hit_tri_v1=reservoir_hits.hit_tri_v1,
                    hit_tri_v2=reservoir_hits.hit_tri_v2,
                    tx_positions=tx_positions,
                    rx_positions=rx_positions,
                    n_hits_per_rx=self.config.n_hits_per_rx,
                    scene=self.scene_ctx.scene,
                    verbose=self.verbose,
                    use_patches=use_patches,
                    hit_ID=reservoir_hits.hit_ID,
                    patch_data=self.patch_data,
                )
                n_valid_spec = int(dr.sum(mi.UInt32(specular_paths.valid))[0]) if specular_paths.n_paths > 0 else 0
                n_total_hits = dedup_stats['n_valid_hits']
                n_unique = dedup_stats['n_unique_triangles']
                n_deduped = dedup_stats['n_duplicates_removed']
                print(f"  [ImageMethod] Diffuse hits: {n_total_hits:,} | "
                      f"Unique {dedup_label}: {n_unique:,} (dedup removed {n_deduped:,}) | "
                      f"Valid specular paths: {n_valid_spec:,}/{specular_paths.n_paths:,}")

                if use_patches and self.patch_data is not None and specular_paths.patch_id is not None:
                    n_patches = self.patch_data.n_patches
                    patch_has_specular_np = np.zeros(n_patches, dtype=bool)
                    valid_sp_np = np.array(specular_paths.valid)
                    patch_id_np = np.array(specular_paths.patch_id)
                    valid_patch_ids = patch_id_np[valid_sp_np].astype(np.int64)
                    valid_patch_ids = valid_patch_ids[(valid_patch_ids >= 0) & (valid_patch_ids < n_patches)]
                    patch_has_specular_np[valid_patch_ids] = True
                    n_covered = int(np.sum(patch_has_specular_np))
                    print(f"  [Per-Patch BSDF] {n_covered}/{n_patches} patches have specular coverage")

                    patch_has_specular = mi.Bool(patch_has_specular_np)
                    tri_to_patch = self.patch_data.tri_to_patch
                    use_non_ka = True
                else:
                    if n_valid_spec > 0:
                        use_non_ka = True
                    else:
                        use_non_ka = False

        # Sync triangle materials to integrator for Phase A specular BSDF evaluation
        self.integrator.triangle_materials = self.triangle_materials

        # ADC synthesis with cached geometry
        if self.config.use_shared_hits:
            result = self.integrator.synthesize_shared_hits_vectorized(
                tx_positions=tx_positions,
                rx_positions=rx_positions,
                tx_boresights=tx_boresights,
                rx_boresights=rx_boresights,
                reservoir_hits=reservoir_hits,
                scene=self.scene_ctx.scene,
                tx_pattern_rrts=self.tx_pattern_rrts,
                rx_pattern_rrts=self.rx_pattern_rrts,
                tx_pattern_loader=self.tx_pattern_loader,
                rx_pattern_loader=self.rx_pattern_loader,
                antenna_gain_linear=self.config.antenna_gain_linear,
                verbose=self.verbose,
                use_non_ka_bsdf=use_non_ka,
                specular_paths=specular_paths,
                patch_has_specular=patch_has_specular,
                tri_to_patch=tri_to_patch,
                return_cached_geometry=True,
            )
            adc_result, cached_geometry = result
        else:
            adc_result = self.integrator.synthesize_vectorized(
                tx_positions=tx_positions,
                rx_positions=rx_positions,
                reservoir_hits=reservoir_hits,
                scene=self.scene_ctx.scene,
                verbose=self.verbose
            )
            cached_geometry = None

        self._last_result = RenderResultRef(
            adc_result=adc_result,
            reservoir_hits=reservoir_hits,
            config=self.config
        )

        return self._last_result, cached_geometry

    def render_differentiable(self, cached_geometry, raw_params_drjit,
                              pose_params=None, normal_params=None,
                              pattern_loaders=None, vertex_offset_params=None):
        """
        Phase B: Differentiable ADC re-synthesis using cached geometry.

        Takes cached geometry from Phase A and grad-enabled parameters,
        returns live DrJit ADC arrays suitable for backward differentiation.

        Supports multiple differentiable parameter groups:
        - raw_params_drjit: Always used. Material parameters in unconstrained space.
        - pose_params: Optional. Dict with 'pitch','roll','yaw','tx','ty','tz' (mi.Float).
        - normal_params: Optional. List of 3 mi.Float arrays [nx, ny, nz] per vertex.
        - pattern_loaders: Optional. Dict {'tx': loader, 'rx': loader} with
            grad-enabled AntennaPatternLoader instances.
        - vertex_offset_params: Optional. List of 3 mi.Float arrays [dx, dy, dz]
            per vertex for mesh refinement.

        Args:
            cached_geometry: CachedGeometry from render_with_cached_geometry().
            raw_params_drjit: list of 6 mi.Float arrays with dr.enable_grad(),
                each of length n_params in unconstrained space.
            pose_params: Optional dict of 6DOF pose parameters.
            normal_params: Optional per-vertex normal parameters.
            pattern_loaders: Optional antenna pattern loaders.
            vertex_offset_params: Optional per-vertex position offsets.

        Returns:
            (adc_real_flat, adc_imag_flat): DrJit mi.Float arrays [n_tx * n_rx * K].
        """
        from .bsdf.mmwave_scalar import reparameterize_physics_params_drjit

        # Reparameterize: unconstrained → bounded physics params (differentiable)
        physics_params = reparameterize_physics_params_drjit(raw_params_drjit)

        # Differentiable synthesis with all parameter groups
        return self.integrator.synthesize_differentiable(
            cached_geometry, physics_params,
            pose_params=pose_params,
            normal_params=normal_params,
            pattern_loaders=pattern_loaders,
            vertex_offset_params=vertex_offset_params,
        )

    def _get_triangle_materials_gpu(self):
        """Return GPU-resident per-triangle material columns (lazy upload).

        Converts the numpy ``triangle_materials`` array to a list of
        ``mi.Float`` columns on the GPU.  The GPU copy is cached and
        invalidated whenever the numpy array object identity changes.
        """
        tm = self.triangle_materials
        if tm is None:
            self._triangle_materials_gpu = None
            return None
        # Re-upload when the numpy array changes (identity check is cheap)
        if (self._triangle_materials_gpu is None
                or self._triangle_materials_gpu_src is not tm):
            cols = []
            for c in range(tm.shape[1]):
                cols.append(mi.Float(tm[:, c].astype(np.float32)))
            dr.eval(*cols)  # single JIT compilation for all columns
            self._triangle_materials_gpu = cols
            self._triangle_materials_gpu_src = tm
        return self._triangle_materials_gpu

    def render_end_to_end(
        self,
        raw_params_drjit: list,
        pose_params: Optional[dict] = None,
        normal_params: Optional[list] = None,
        vertex_offset_params: Optional[list] = None,
        pattern_loaders: Optional[dict] = None,
        seed: Optional[int] = None,
        include_diffuse: bool = True,
        include_specular: bool = True,
        include_diffraction: bool = True,
        include_energy_borrowing: bool = True,
        return_total_distance: bool = False,
    ) -> tuple:
        """
        Single-pass end-to-end differentiable render.

        All operations (ray tracing, BSDF evaluation, phase computation,
        ADC accumulation) are in one DrJit AD graph. Gradients flow through
        vertex positions, normals, materials, pose, and antenna patterns.

        Supports both diffuse MC paths and SMS specular paths in the same
        AD graph. Use include_diffuse/include_specular/include_diffraction
        to control which contributions are included (useful for isolating
        gradient sources in testing).

        Args:
            raw_params_drjit: list of 6 mi.Float arrays with dr.enable_grad(),
                each of length n_params in unconstrained (reparameterized) space.
            pose_params: Optional dict with 'pitch','roll','yaw','tx','ty','tz'
                (mi.Float scalars). Transforms TX/RX positions differentiably.
            normal_params: Optional list of 3 mi.Float arrays [nx, ny, nz] per vertex.
            vertex_offset_params: Optional list of 3 mi.Float arrays [dx, dy, dz]
                per vertex. Modifies scene geometry for AD-attached intersection.
            pattern_loaders: Optional dict {'tx': loader, 'rx': loader}.
            seed: Random seed (uses config seed if None).
            include_diffuse: If True, include diffuse MC path contributions.
            include_specular: If True, include SMS specular path contributions.
            include_diffraction: If True, include FSD-BSDF diffraction contributions.
            return_total_distance: If True, also return total path distance sum
                (mi.Float scalar) for phase-only gradient testing.

        Returns:
            (adc_real_flat, adc_imag_flat): DrJit mi.Float arrays [n_tx * n_rx * K]
                with gradients attached through the full computation graph.
            If return_total_distance=True:
            (adc_real_flat, adc_imag_flat, total_distance): also returns mi.Float
                scalar with sum of all path distances (d_tx + d_rx) for phase testing.
        """
        from .bsdf.mmwave_scalar import reparameterize_physics_params_drjit
        import time as _time

        seed = seed or self.config.seed



        # Profiling support
        _prof = getattr(self.integrator, '_profile', None)
        def _tic(label):
            if _prof is not None:
                dr.sync_thread()
                _prof[label] = -_time.perf_counter()
        def _toc(label):
            if _prof is not None:
                dr.sync_thread()
                _prof[label] += _time.perf_counter()

        _tic('material_reparam')
        # 0. Reparameterize materials (diff)
        physics_params = reparameterize_physics_params_drjit(raw_params_drjit)
        _toc('material_reparam')

        _tic('antenna_pose')
        # 1. Apply pose to TX/RX positions (diff if pose_params provided)
        tx_positions, rx_positions, tx_boresights, rx_boresights = \
            self._get_antenna_arrays(pose_params)
        _toc('antenna_pose')

        # 2. Update scene vertex positions if vertex offsets provided
        if vertex_offset_params is not None:
            self._update_scene_vertices(vertex_offset_params)

        _tic('ray_sampling')
        # 3. Sample rays per RX (non-diff directions, AD-attached origins from RX)
        reservoir_hits = self.sampler.sample_reservoir_drjit(
            scene=self.scene_ctx.scene,
            rx_positions=rx_positions,
            rx_boresights=rx_boresights,
            seed=seed,
            verbose=self.verbose,
            use_vertex_normals=self.config.use_vertex_normals,
        )
        _toc('ray_sampling')

        _tic('sms_specular_cache')
        # 3b. Find SMS specular paths on the (possibly deformed) mesh.
        # SMS paths are cached to avoid re-running the expensive Newton solver
        # every iteration. IFT in synthesis handles geometry perturbations.
        #
        # Caching strategy:
        # - Static geometry (no pose/vtx learning): cache forever
        # - Dynamic geometry with sms_cache_interval > 0: cache for N iters
        # - Dynamic geometry with sms_cache_interval == 0: recompute every iter
        _geom_static = (pose_params is None and vertex_offset_params is None)
        _cache_interval = self.config.sms_cache_interval

        # Initialize cache state
        if not hasattr(self, '_sms_cache_iter'):
            self._sms_cache_iter = 0
            self._cached_specular_paths = None
            self._cached_specular_chains = None

        # Determine if SMS needs refresh
        _need_refresh = False
        if _geom_static:
            _need_refresh = (self._cached_specular_paths is None)
        elif _cache_interval > 0:
            _need_refresh = (self._cached_specular_paths is None
                             or self._sms_cache_iter >= _cache_interval)
        else:
            _need_refresh = True  # recompute every iteration

        specular_paths = None
        if include_specular and self.sms_solver is not None:
            if _need_refresh:
                self._cached_specular_paths = self._find_sms_specular_e2e(
                    reservoir_hits, tx_positions, rx_positions,
                    self.scene_ctx.scene)
                self._sms_cache_iter = 0
            specular_paths = self._cached_specular_paths

        # 3c. Find multibounce specular chains (k >= 2)
        specular_chains = None
        if (include_specular and self.sms_solver is not None
                and self.config.max_bounces > 1):
            if _need_refresh:
                self._cached_specular_chains = self._find_sms_specular_chains_e2e(
                    reservoir_hits, tx_positions, rx_positions,
                    self.scene_ctx.scene,
                    k=min(self.config.max_bounces, 2),
                    seed=seed,
                )
            specular_chains = self._cached_specular_chains

        self._sms_cache_iter += 1

        _toc('sms_specular_cache')

        # 4. Pass vertex position buffer to integrator for diff re-intersection
        if hasattr(self, '_scene_params') and hasattr(self, '_vp_key'):
            self.integrator._scene_params = self._scene_params
            self.integrator._vp_key = self._vp_key

        # 5. End-to-end synthesis (single-bounce or multibounce)
        if self.config.max_bounces > 1:
            synth_result = self.integrator.synthesize_end_to_end_multibounce(
                reservoir_hits=reservoir_hits,
                scene=self.scene_ctx.scene,
                tx_positions=tx_positions,
                rx_positions=rx_positions,
                tx_boresights=tx_boresights,
                rx_boresights=rx_boresights,
                physics_params=physics_params,
                normal_params=normal_params,
                pattern_loaders=pattern_loaders,
                tx_pattern_rrts=self.tx_pattern_rrts,
                rx_pattern_rrts=self.rx_pattern_rrts,
                tx_pattern_loader=self.tx_pattern_loader,
                rx_pattern_loader=self.rx_pattern_loader,
                antenna_gain_linear=self.config.antenna_gain_linear,
                verbose=self.verbose,
                specular_paths=specular_paths,
                specular_chains=specular_chains,
                include_diffuse=include_diffuse,
                include_specular=include_specular,
                include_diffraction=include_diffraction,
                include_energy_borrowing=include_energy_borrowing,
                vertex_offset_params=vertex_offset_params,
                return_total_distance=return_total_distance,
                max_bounces=self.config.max_bounces,
                nee_every_bounce=self.config.nee_every_bounce,
                rr_start_bounce=self.config.rr_start_bounce,
                rr_prob=self.config.rr_prob,
                compute_boundary_viewpoints=(
                    self.boundary_computer is not None
                    and self.boundary_computer.enabled
                    and (vertex_offset_params is not None or pose_params is not None)
                ),
            )
        else:
            synth_result = self.integrator.synthesize_end_to_end(
                reservoir_hits=reservoir_hits,
                scene=self.scene_ctx.scene,
                tx_positions=tx_positions,
                rx_positions=rx_positions,
                tx_boresights=tx_boresights,
                rx_boresights=rx_boresights,
                physics_params=physics_params,
                normal_params=normal_params,
                pattern_loaders=pattern_loaders,
                tx_pattern_rrts=self.tx_pattern_rrts,
                rx_pattern_rrts=self.rx_pattern_rrts,
                tx_pattern_loader=self.tx_pattern_loader,
                rx_pattern_loader=self.rx_pattern_loader,
                antenna_gain_linear=self.config.antenna_gain_linear,
                verbose=self.verbose,
                specular_paths=specular_paths,
                include_diffuse=include_diffuse,
                include_specular=include_specular,
                include_diffraction=include_diffraction,
                include_energy_borrowing=include_energy_borrowing,
                vertex_offset_params=vertex_offset_params,
                return_total_distance=return_total_distance,
            )

        if return_total_distance:
            adc_real, adc_imag, total_distance = synth_result
        else:
            adc_real, adc_imag = synth_result

        # 6. Boundary gradient correction (optional)
        # Adds the missing visibility discontinuity term at silhouette edges.
        # Only active when geometry parameters are being optimized.
        if (self.boundary_computer is not None
                and self.boundary_computer.enabled
                and (vertex_offset_params is not None or pose_params is not None)):

            if self.config.max_bounces > 1:
                # Multibounce: compute boundary gradients at each bounce level.
                # Per-bounce viewpoints are stored by the integrator during synthesis.
                per_bounce_viewpoints = getattr(
                    self.integrator, '_multibounce_viewpoints', [])
                if per_bounce_viewpoints:
                    self.boundary_computer.compute_boundary_gradients_multibounce(
                        adc_real_flat=adc_real,
                        adc_imag_flat=adc_imag,
                        scene=self.scene_ctx.scene,
                        tx_positions=tx_positions,
                        rx_positions=rx_positions,
                        per_bounce_viewpoints=per_bounce_viewpoints,
                        vertex_offset_params=vertex_offset_params,
                        seed=seed,
                        verbose=self.verbose,
                    )
            else:
                # Single-bounce: original behavior
                # Initialize silhouettes on first call (needs TX centroid)
                if not self.boundary_computer.has_silhouettes:
                    tx_centroid = mi.ScalarPoint3f(
                        float(dr.mean(tx_positions.x)[0]),
                        float(dr.mean(tx_positions.y)[0]),
                        float(dr.mean(tx_positions.z)[0]),
                    )
                    self.boundary_computer.init_silhouettes(
                        self.scene_ctx.scene, tx_centroid)

                if self.boundary_computer.has_silhouettes:
                    self.boundary_computer.compute_boundary_gradients(
                        adc_real_flat=adc_real,
                        adc_imag_flat=adc_imag,
                        scene=self.scene_ctx.scene,
                        tx_positions=tx_positions,
                        rx_positions=rx_positions,
                        vertex_offset_params=vertex_offset_params,
                        seed=seed,
                        verbose=self.verbose,
                    )

        # Note: Do NOT auto-restore scene vertices here. The AD tape for
        # vertex position gradients requires the modified positions to remain
        # in the scene until dr.backward() completes. Callers should call
        # _restore_scene_vertices() after backward if needed (or the next
        # _update_scene_vertices call will overwrite them anyway).

        if return_total_distance:
            return adc_real, adc_imag, total_distance
        return adc_real, adc_imag

    def _find_sms_specular_e2e(self, reservoir_hits, tx_positions, rx_positions, scene):
        """
        Find SMS specular paths from e2e reservoir hits.

        Uses plane-hash deduplication (matching the sionna reference renderer)
        to collapse co-planar triangles per-RX, then calls the SMS solver to
        find specular reflection points via Newton iteration.

        The SMS solver runs WITHOUT AD (forward Newton iterations only).
        Gradients are attached retroactively via IFT in _synthesize_specular_e2e().

        Args:
            reservoir_hits: ReservoirHitsDrJit from sample_reservoir_drjit
            tx_positions: mi.Point3f [n_tx] (possibly pose-transformed)
            rx_positions: mi.Point3f [n_rx] (possibly pose-transformed)
            scene: mi.Scene (possibly with deformed vertices)

        Returns:
            SpecularPaths or None if no valid specular paths found.
        """
        n_valid = reservoir_hits.n_valid
        if n_valid == 0:
            return None

        # Dedup on compressed hits (per-RX). Uses patch dedup when available
        # (matches sionna reference renderer exactly), else plane-hash fallback.
        if self.deduplicator is not None:
            unique_mask, dedup_stats = self.deduplicator.deduplicate_compressed(
                hit_N=reservoir_hits.hit_N,
                hit_P=reservoir_hits.hit_P,
                rx_idx=reservoir_hits.rx_element_idx,
                n_rx=reservoir_hits.n_rx,
                hit_prim_ids=reservoir_hits.hit_prim_ids,
                patch_data=self.patch_data,
            )
            dedup_mode = "Patch" if dedup_stats.get('use_patches') else "Plane-hash"
            if self.verbose:
                print(f"  [SMS-E2E] {dedup_mode} dedup: {dedup_stats['n_unique_triangles']:,} unique "
                      f"/ {dedup_stats['n_valid_hits']:,} valid "
                      f"({dedup_stats['n_duplicates_removed']:,} duplicates removed)")
            # Compress unique hits to seed indices
            unique_indices = dr.compress(unique_mask)
            n_unique = dr.width(unique_indices)
        else:
            # GPU-native unique via scatter_reduce(Min)
            prim_ids = reservoir_hits.hit_prim_ids
            n_hits = dr.width(prim_ids)
            hit_indices = dr.arange(mi.UInt32, n_hits)
            dr.eval(prim_ids)
            max_pid = int(dr.max(prim_ids)[0])
            buf_size = max_pid + 1
            first_occ = dr.full(mi.UInt32, 0xFFFFFFFF, buf_size)
            dr.scatter_reduce(dr.ReduceOp.Min, first_occ, hit_indices, prim_ids)
            valid_tris = dr.compress(first_occ != mi.UInt32(0xFFFFFFFF))
            unique_indices = dr.gather(mi.UInt32, first_occ, valid_tris)
            n_unique = dr.width(unique_indices)

        if n_unique == 0:
            return None

        # Gather seed data at unique occurrences.
        # Detach from AD — SMS solver runs without AD (IFT attaches grads later).
        # PERF: Gather all components first, then evaluate in a single dr.eval()
        # call to avoid 6 separate JIT compilations of the AD-attached ray
        # intersection graph (each dr.detach() on a lazy AD var forces eval).
        _px = dr.gather(mi.Float, reservoir_hits.hit_P.x, unique_indices)
        _py = dr.gather(mi.Float, reservoir_hits.hit_P.y, unique_indices)
        _pz = dr.gather(mi.Float, reservoir_hits.hit_P.z, unique_indices)
        _nx = dr.gather(mi.Float, reservoir_hits.hit_N.x, unique_indices)
        _ny = dr.gather(mi.Float, reservoir_hits.hit_N.y, unique_indices)
        _nz = dr.gather(mi.Float, reservoir_hits.hit_N.z, unique_indices)
        dr.eval(_px, _py, _pz, _nx, _ny, _nz)
        seed_P = mi.Point3f(mi.Float(dr.detach(_px)), mi.Float(dr.detach(_py)), mi.Float(dr.detach(_pz)))
        seed_N = mi.Vector3f(mi.Float(dr.detach(_nx)), mi.Float(dr.detach(_ny)), mi.Float(dr.detach(_nz)))
        seed_prim_ids = dr.gather(mi.UInt32, reservoir_hits.hit_prim_ids, unique_indices)
        seed_rx_idx = dr.gather(mi.UInt32, reservoir_hits.rx_element_idx, unique_indices)

        # Detach TX/RX positions from AD graph — SMS Newton solver runs
        # without AD. Gradients are attached retroactively via IFT in
        # _synthesize_specular_e2e(). Without detach, the Newton iterations
        # create an enormous AD tape that hangs backward traversal.
        tx_pos_detached = mi.Point3f(
            mi.Float(dr.detach(tx_positions.x)),
            mi.Float(dr.detach(tx_positions.y)),
            mi.Float(dr.detach(tx_positions.z)),
        )
        rx_pos_detached = mi.Point3f(
            mi.Float(dr.detach(rx_positions.x)),
            mi.Float(dr.detach(rx_positions.y)),
            mi.Float(dr.detach(rx_positions.z)),
        )

        # Wrap in dr.suspend_grad() to prevent the SMS Newton solver from
        # recording AD operations through the scene's vertex buffer. When
        # vertex_offset_params are used, the scene's vertices are AD-attached,
        # and the solver's ray_test() / surface_parameterization() calls would
        # create an enormous AD tape that hangs backward traversal.
        # IFT in _synthesize_specular_e2e() attaches gradients retroactively.
        with dr.suspend_grad():
            return self.sms_solver.find_specular_paths_from_seeds(
                seed_P=seed_P,
                seed_N=seed_N,
                seed_prim_ids=seed_prim_ids,
                seed_rx_idx=seed_rx_idx,
                tx_positions=tx_pos_detached,
                rx_positions=rx_pos_detached,
                scene=scene,
                verbose=self.verbose,
                triangle_materials_gpu=self._get_triangle_materials_gpu(),
                n_material_cols=self.triangle_materials.shape[1] if self.triangle_materials is not None else 0,
                patch_data=self.patch_data,
            )

    def _find_sms_specular_chains_e2e(
        self, reservoir_hits, tx_positions, rx_positions, scene, k=2, seed=42,
    ):
        """
        Find k-bounce specular chains via multibounce SMS.

        Seeding strategy (Option B: Random Walk):
        1. Take dedup'd reservoir hits as S₁ seeds
        2. From each S₁, trace cosine-weighted random rays to find S₂ candidates
        3. Run block-tridiagonal Newton solver on (S₁, S₂) pairs

        Args:
            reservoir_hits: ReservoirHitsDrJit from sample_reservoir_drjit
            tx_positions: mi.Point3f [n_tx]
            rx_positions: mi.Point3f [n_rx]
            scene: mi.Scene
            k: Number of specular bounces (default 2)
            seed: Random seed for S₂ sampling

        Returns:
            MultibounceSpecularChain or None
        """
        from .specular.sms import MultibounceSpecularChain
        from .integrator import _sample_cosine_hemisphere_drjit

        n_valid = reservoir_hits.n_valid
        if n_valid == 0:
            return None

        n_tx = dr.width(tx_positions)

        # Dedup: use GPU deduplicator if available, else GPU-native fallback
        if self.deduplicator is not None:
            unique_mask, _dedup_stats = self.deduplicator.deduplicate_compressed(
                hit_N=reservoir_hits.hit_N,
                hit_P=reservoir_hits.hit_P,
                rx_idx=reservoir_hits.rx_element_idx,
                n_rx=reservoir_hits.n_rx,
                hit_prim_ids=reservoir_hits.hit_prim_ids,
                patch_data=self.patch_data,
            )
            idx_dr = dr.compress(unique_mask)
            n_unique = dr.width(idx_dr)
        else:
            # GPU-native unique via scatter_reduce(Min): for each prim_id,
            # store the minimum hit index. Then compress non-empty entries.
            prim_ids = reservoir_hits.hit_prim_ids
            n_hits = dr.width(prim_ids)
            hit_indices = dr.arange(mi.UInt32, n_hits)
            # Need buffer size = max_prim_id + 1
            dr.eval(prim_ids)
            max_pid = int(dr.max(prim_ids)[0])
            buf_size = max_pid + 1
            first_occ = dr.full(mi.UInt32, 0xFFFFFFFF, buf_size)
            dr.scatter_reduce(dr.ReduceOp.Min, first_occ, hit_indices, prim_ids)
            # Compress to get triangle IDs that were hit
            valid_tris = dr.compress(first_occ != mi.UInt32(0xFFFFFFFF))
            # Gather the original hit indices for those triangles
            idx_dr = dr.gather(mi.UInt32, first_occ, valid_tris)
            n_unique = dr.width(idx_dr)

        if n_unique == 0:
            return None

        # Gather S₁ seed data (detached from AD)
        # PERF: batch all gathers, single dr.eval(), then detach
        _px = dr.gather(mi.Float, reservoir_hits.hit_P.x, idx_dr)
        _py = dr.gather(mi.Float, reservoir_hits.hit_P.y, idx_dr)
        _pz = dr.gather(mi.Float, reservoir_hits.hit_P.z, idx_dr)
        _nx = dr.gather(mi.Float, reservoir_hits.hit_N.x, idx_dr)
        _ny = dr.gather(mi.Float, reservoir_hits.hit_N.y, idx_dr)
        _nz = dr.gather(mi.Float, reservoir_hits.hit_N.z, idx_dr)
        s1_prim_ids = dr.gather(mi.UInt32, reservoir_hits.hit_prim_ids, idx_dr)
        s1_rx_idx = dr.gather(mi.UInt32, reservoir_hits.rx_element_idx, idx_dr)
        dr.eval(_px, _py, _pz, _nx, _ny, _nz, s1_prim_ids, s1_rx_idx)
        s1_P = mi.Point3f(mi.Float(dr.detach(_px)), mi.Float(dr.detach(_py)), mi.Float(dr.detach(_pz)))
        s1_N = mi.Vector3f(mi.Float(dr.detach(_nx)), mi.Float(dr.detach(_ny)), mi.Float(dr.detach(_nz)))
        s1_prim_ids = mi.UInt32(dr.detach(s1_prim_ids))
        s1_rx_idx = mi.UInt32(dr.detach(s1_rx_idx))

        # Expand × TX: n_chains = n_unique × n_tx
        n_chains = n_unique * n_tx

        s1_P_exp = mi.Point3f(
            dr.repeat(s1_P.x, n_tx), dr.repeat(s1_P.y, n_tx), dr.repeat(s1_P.z, n_tx))
        s1_N_exp = mi.Vector3f(
            dr.repeat(s1_N.x, n_tx), dr.repeat(s1_N.y, n_tx), dr.repeat(s1_N.z, n_tx))
        s1_prim_exp = mi.UInt32(dr.repeat(s1_prim_ids, n_tx))

        # PERF: GPU-native tiling instead of np.tile/np.repeat
        # chain_tx_idx: [0,1,...,n_tx-1, 0,1,...,n_tx-1, ...] repeated n_unique times
        chain_tx_idx = dr.arange(mi.UInt32, n_chains) % mi.UInt32(n_tx)
        # chain_rx_idx: each s1_rx_idx repeated n_tx times
        chain_rx_idx = mi.UInt32(dr.repeat(s1_rx_idx, n_tx))

        # Generate S₂ seeds: trace random ray from S₁ (cosine-weighted)
        with dr.suspend_grad():
            s2_dirs, _s2_pdf, _s2_sampler = _sample_cosine_hemisphere_drjit(s1_N_exp, n_chains, seed + 12345)

            # Offset origin along normal to avoid self-intersection
            offset = mi.Float(1e-3)
            ray_o = mi.Point3f(
                s1_P_exp.x + offset * s1_N_exp.x,
                s1_P_exp.y + offset * s1_N_exp.y,
                s1_P_exp.z + offset * s1_N_exp.z)
            ray = mi.Ray3f(ray_o, s2_dirs)
            si = scene.ray_intersect(ray)

            s2_P = mi.Point3f(si.p)
            s2_prim_ids = mi.UInt32(si.prim_index)
            s2_valid = si.is_valid()

            # Filter out chains where S₂ didn't hit anything
            n_s2_valid = int(dr.sum(mi.UInt32(s2_valid))[0])

            if self.verbose:
                print(f"  [SMS-MB] S₂ seeding: {n_s2_valid}/{n_chains} hit surface")

            if n_s2_valid == 0:
                return None

            # Build seed lists for find_specular_chains
            seed_P_list = [s1_P_exp, s2_P]
            seed_prim_list = [s1_prim_exp, s2_prim_ids]

            # For k > 2, would add more bounce seeds here (future extension)

            result = self.sms_solver.find_specular_chains(
                seed_P_list=seed_P_list,
                seed_prim_ids_list=seed_prim_list,
                tx_positions=mi.Point3f(
                    mi.Float(dr.detach(tx_positions.x)),
                    mi.Float(dr.detach(tx_positions.y)),
                    mi.Float(dr.detach(tx_positions.z))),
                rx_positions=mi.Point3f(
                    mi.Float(dr.detach(rx_positions.x)),
                    mi.Float(dr.detach(rx_positions.y)),
                    mi.Float(dr.detach(rx_positions.z))),
                seed_tx_idx=chain_tx_idx,
                seed_rx_idx=chain_rx_idx,
                scene=scene,
                k=k,
                max_iterations=self.config.sms_max_iterations,
                threshold=self.config.sms_solver_threshold,
                verbose=self.verbose,
                triangle_materials_gpu=self._get_triangle_materials_gpu(),
                n_material_cols=self.triangle_materials.shape[1] if self.triangle_materials is not None else 0,
            )

        return result

    def _get_antenna_arrays(self, pose_params=None):
        """
        Return TX/RX positions and boresights, optionally transformed by pose.

        All operations in DrJit for AD continuity when pose_params are grad-enabled.

        Returns:
            (tx_positions, rx_positions, tx_boresights, rx_boresights)
        """
        tx_array = self.scene_ctx.tx_array
        rx_array = self.scene_ctx.rx_array

        tx_positions = mi.Point3f(
            mi.Float(tx_array.positions[:, 0]),
            mi.Float(tx_array.positions[:, 1]),
            mi.Float(tx_array.positions[:, 2]),
        )
        rx_positions = mi.Point3f(
            mi.Float(rx_array.positions[:, 0]),
            mi.Float(rx_array.positions[:, 1]),
            mi.Float(rx_array.positions[:, 2]),
        )

        n_tx = tx_array.num_elements
        n_rx = rx_array.num_elements

        if hasattr(tx_array, 'orientations') and tx_array.orientations is not None:
            tx_boresights = tx_array.orientations
        else:
            tx_boresights = mi.Vector3f(
                mi.Float(np.zeros(n_tx)), mi.Float(np.ones(n_tx)), mi.Float(np.zeros(n_tx)))

        if hasattr(rx_array, 'orientations') and rx_array.orientations is not None:
            rx_boresights = rx_array.orientations
        else:
            rx_boresights = mi.Vector3f(
                mi.Float(np.zeros(n_rx)), mi.Float(np.ones(n_rx)), mi.Float(np.zeros(n_rx)))

        if pose_params is not None:
            from .utils.transforms import transform_positions
            tx_positions = transform_positions(tx_positions, pose_params)
            rx_positions = transform_positions(rx_positions, pose_params)
            # Boresights also rotate with the pose
            tx_boresights = transform_positions(
                mi.Point3f(tx_boresights.x, tx_boresights.y, tx_boresights.z),
                pose_params, translate=False)
            tx_boresights = mi.Vector3f(tx_boresights.x, tx_boresights.y, tx_boresights.z)
            rx_boresights = transform_positions(
                mi.Point3f(rx_boresights.x, rx_boresights.y, rx_boresights.z),
                pose_params, translate=False)
            rx_boresights = mi.Vector3f(rx_boresights.x, rx_boresights.y, rx_boresights.z)

        return tx_positions, rx_positions, tx_boresights, rx_boresights

    def _update_scene_vertices(self, vertex_offset_params):
        """
        Apply vertex offsets to the Mitsuba scene mesh for AD-attached intersection.

        vertex_offset_params: [dx, dy, dz] — 3 mi.Float arrays, grad-enabled

        Uses gather+select (not scatter) to build the interleaved array so that
        DrJit AD properly tracks gradients from dx/dy/dz through to vertex_positions.
        """
        scene = self.scene_ctx.scene

        # Store base positions on first call (traverse scene for proper AD hooks)
        if not hasattr(self, '_scene_params'):
            self._scene_params = mi.traverse(scene)
            # Find vertex_positions key (may be prefixed with mesh name)
            vp_key = None
            for k in self._scene_params.keys():
                if k.endswith('vertex_positions'):
                    vp_key = k
                    break
            self._vp_key = vp_key or 'vertex_positions'
            self._base_vertex_positions = mi.Float(self._scene_params[self._vp_key])
            dr.eval(self._base_vertex_positions)

        params = self._scene_params
        base = self._base_vertex_positions
        dx, dy, dz = vertex_offset_params

        n_verts = dr.width(dx)
        n_flat = n_verts * 3

        # Build interleaved offset using gather+select (AD-friendly)
        # Layout: [x0, y0, z0, x1, y1, z1, ...]
        flat_idx = dr.arange(mi.UInt32, n_flat)
        component = flat_idx % mi.UInt32(3)   # 0,1,2,0,1,2,...
        vert_idx = flat_idx // mi.UInt32(3)   # 0,0,0,1,1,1,...

        # Gather from each component (AD flows through dx/dy/dz)
        ox = dr.gather(mi.Float, dx, vert_idx)
        oy = dr.gather(mi.Float, dy, vert_idx)
        oz = dr.gather(mi.Float, dz, vert_idx)

        # Select the right component per flat index
        offset_flat = dr.select(
            component == mi.UInt32(0), ox,
            dr.select(component == mi.UInt32(1), oy, oz))

        new_positions = base + offset_flat
        params[self._vp_key] = new_positions
        params.update()  # BVH rebuild (~11ms for 50K verts)

    def _restore_scene_vertices(self):
        """Restore scene vertices to base positions (undo offsets)."""
        if hasattr(self, '_scene_params') and hasattr(self, '_base_vertex_positions'):
            self._scene_params[self._vp_key] = mi.Float(self._base_vertex_positions)
            self._scene_params.update()

    def render_with_components(self, seed: Optional[int] = None):
        """
        Execute rendering and return individual BSDF component ADC arrays.

        This method is useful for visualizing/analyzing individual BSDF components:
        - KA (Kirchhoff Approximation) specular lobe
        - SPM (Small Perturbation Method) diffuse lobe
        - Directive incoherent lobe
        - Broad diffuse lobe
        - Coherent (combined KA + SPM)
        - Incoherent (combined directive + broad)

        Note: Only works with mmwave_scalar or mmwave_jones BSDF models.

        Args:
            seed: Random seed (uses config seed if None)

        Returns:
            ADCComponentResult with component ADC arrays, or None if BSDF
            doesn't support component extraction
        """
        from .integrator import ADCComponentResult

        seed = seed or self.config.seed

        if self.verbose:
            print("\n" + "=" * 60)
            print("RRTS Reference Renderer (COMPONENT MODE)")
            print("=" * 60)

        # Get RX positions and boresights
        rx_array = self.scene_ctx.rx_array
        n_rx = rx_array.num_elements

        rx_positions = mi.Point3f(
            mi.Float(rx_array.positions[:, 0]),
            mi.Float(rx_array.positions[:, 1]),
            mi.Float(rx_array.positions[:, 2])
        )

        if hasattr(rx_array, 'orientations') and rx_array.orientations is not None:
            rx_boresights = rx_array.orientations
        else:
            rx_boresights = mi.Vector3f(
                mi.Float(np.zeros(n_rx)),
                mi.Float(np.ones(n_rx)),
                mi.Float(np.zeros(n_rx))
            )

        # Reservoir sampling
        reservoir_hits = self.sampler.sample_vectorized(
            scene=self.scene_ctx.scene,
            rx_positions=rx_positions,
            rx_boresights=rx_boresights,
            seed=seed,
            verbose=self.verbose,
            triangle_materials=self.triangle_materials,
            use_vertex_normals=self.config.use_vertex_normals,
        )

        # Get TX positions and boresights
        tx_array = self.scene_ctx.tx_array
        n_tx = tx_array.num_elements

        tx_positions = mi.Point3f(
            mi.Float(tx_array.positions[:, 0]),
            mi.Float(tx_array.positions[:, 1]),
            mi.Float(tx_array.positions[:, 2])
        )

        if hasattr(tx_array, 'orientations') and tx_array.orientations is not None:
            tx_boresights = tx_array.orientations
        else:
            tx_boresights = mi.Vector3f(
                mi.Float(np.zeros(n_tx)),
                mi.Float(np.ones(n_tx)),
                mi.Float(np.zeros(n_tx))
            )

        # Specular path pipeline (SMS or image method, if enabled)
        specular_paths = None
        use_non_ka = False
        patch_has_specular = None
        tri_to_patch = None

        _specular_enabled = (self.config.enable_sms or self.config.enable_image_method) and self.deduplicator is not None
        if _specular_enabled:
            use_patches = self.patch_data is not None
            method_name = "SMS" if self.config.enable_sms else "Image Method"
            mode_str = "PATCH" if use_patches else "TRIANGLE"

            if self.verbose:
                print(f"\n[{method_name} Pipeline] ({mode_str} mode)")

            unique_mask, dedup_stats = self.deduplicator.deduplicate(
                hit_N=reservoir_hits.hit_N,
                hit_P=reservoir_hits.hit_P,
                valid=reservoir_hits.valid,
                n_rx=rx_array.num_elements,
                use_patches=use_patches,
                hit_ID=reservoir_hits.hit_ID if use_patches else None,
                patch_data=self.patch_data,
            )

            dedup_label = "patches" if dedup_stats.get('use_patches') else "triangles"
            if self.verbose:
                print(f"  Unique {dedup_label}: {dedup_stats['n_unique_triangles']:,} / {dedup_stats['n_valid_hits']:,}")

            if self.config.enable_sms and self.sms_solver is not None:
                # ---- SMS path ----
                specular_paths = self.sms_solver.find_specular_paths(
                    unique_mask=unique_mask,
                    hit_P=reservoir_hits.hit_P,
                    hit_N=reservoir_hits.hit_N,
                    hit_rho=reservoir_hits.hit_rho,
                    hit_ID=reservoir_hits.hit_ID,
                    tx_positions=tx_positions,
                    rx_positions=rx_positions,
                    n_hits_per_rx=self.config.n_hits_per_rx,
                    scene=self.scene_ctx.scene,
                    verbose=self.verbose,
                    patch_data=self.patch_data,
                    triangle_materials=self.triangle_materials,
                )
                n_valid_spec = int(dr.sum(mi.UInt32(specular_paths.valid))[0]) if specular_paths.n_paths > 0 else 0
                print(f"  [SMS] Valid specular paths: {n_valid_spec:,}/{specular_paths.n_paths:,}")

                # Per-patch adaptive BSDF: build mask of which patches have specular coverage
                if use_patches and self.patch_data is not None and specular_paths.patch_id is not None and n_valid_spec > 0:
                    n_patches = self.patch_data.n_patches
                    patch_has_specular_np = np.zeros(n_patches, dtype=bool)
                    valid_sp_np = np.array(specular_paths.valid)
                    patch_id_np = np.array(specular_paths.patch_id)
                    valid_patch_ids = patch_id_np[valid_sp_np].astype(np.int64)
                    valid_patch_ids = valid_patch_ids[(valid_patch_ids >= 0) & (valid_patch_ids < n_patches)]
                    patch_has_specular_np[valid_patch_ids] = True
                    n_covered = int(np.sum(patch_has_specular_np))
                    print(f"  [Per-Patch BSDF] {n_covered}/{n_patches} patches have specular coverage")

                    patch_has_specular = mi.Bool(patch_has_specular_np)
                    tri_to_patch = self.patch_data.tri_to_patch
                    use_non_ka = True
                else:
                    if n_valid_spec > 0:
                        use_non_ka = True
                    else:
                        use_non_ka = False
                        print(f"  [SMS] 0 valid specular paths → using full BSDF (KA retained in MC)")

            elif reservoir_hits.hit_tri_v0 is not None:
                # ---- Image method path ----
                specular_paths = self.image_refiner.refine(
                    unique_mask=unique_mask,
                    hit_P=reservoir_hits.hit_P,
                    hit_N=reservoir_hits.hit_N,
                    hit_rho=reservoir_hits.hit_rho,
                    hit_tri_v0=reservoir_hits.hit_tri_v0,
                    hit_tri_v1=reservoir_hits.hit_tri_v1,
                    hit_tri_v2=reservoir_hits.hit_tri_v2,
                    tx_positions=tx_positions,
                    rx_positions=rx_positions,
                    n_hits_per_rx=self.config.n_hits_per_rx,
                    scene=self.scene_ctx.scene,
                    verbose=self.verbose,
                    use_patches=use_patches,
                    hit_ID=reservoir_hits.hit_ID,
                    patch_data=self.patch_data,
                )
                # Always print image method stats
                n_valid_spec = int(dr.sum(mi.UInt32(specular_paths.valid))[0]) if specular_paths.n_paths > 0 else 0
                n_total_hits = dedup_stats['n_valid_hits']
                n_unique = dedup_stats['n_unique_triangles']
                n_deduped = dedup_stats['n_duplicates_removed']
                print(f"  [ImageMethod] Diffuse hits: {n_total_hits:,} | "
                      f"Unique {dedup_label}: {n_unique:,} (dedup removed {n_deduped:,}) | "
                      f"Valid specular paths: {n_valid_spec:,}/{specular_paths.n_paths:,}")

                # Per-patch adaptive BSDF: build mask of which patches have specular coverage
                if use_patches and self.patch_data is not None and specular_paths.patch_id is not None:
                    n_patches = self.patch_data.n_patches
                    patch_has_specular_np = np.zeros(n_patches, dtype=bool)
                    valid_sp_np = np.array(specular_paths.valid)
                    patch_id_np = np.array(specular_paths.patch_id)
                    valid_patch_ids = patch_id_np[valid_sp_np].astype(np.int64)
                    valid_patch_ids = valid_patch_ids[(valid_patch_ids >= 0) & (valid_patch_ids < n_patches)]
                    patch_has_specular_np[valid_patch_ids] = True
                    n_covered = int(np.sum(patch_has_specular_np))
                    print(f"  [Per-Patch BSDF] {n_covered}/{n_patches} patches have specular coverage")

                    patch_has_specular = mi.Bool(patch_has_specular_np)
                    tri_to_patch = self.patch_data.tri_to_patch
                    use_non_ka = True
                else:
                    if n_valid_spec > 0:
                        use_non_ka = True
                    else:
                        use_non_ka = False
                        print(f"  [Adaptive BSDF] 0 valid specular paths → using full BSDF (KA retained in MC)")

        # Sync triangle materials to integrator for Phase A specular BSDF evaluation
        self.integrator.triangle_materials = self.triangle_materials

        # ADC synthesis with components
        if self.config.use_shared_hits:
            result = self.integrator.synthesize_shared_hits_vectorized(
                tx_positions=tx_positions,
                rx_positions=rx_positions,
                tx_boresights=tx_boresights,
                rx_boresights=rx_boresights,
                reservoir_hits=reservoir_hits,
                scene=self.scene_ctx.scene,
                tx_pattern_rrts=self.tx_pattern_rrts,
                rx_pattern_rrts=self.rx_pattern_rrts,
                tx_pattern_loader=self.tx_pattern_loader,
                rx_pattern_loader=self.rx_pattern_loader,
                antenna_gain_linear=self.config.antenna_gain_linear,
                verbose=self.verbose,
                return_components=True,  # Request component-level rendering
                use_non_ka_bsdf=use_non_ka,
                specular_paths=specular_paths,
                patch_has_specular=patch_has_specular,
                tri_to_patch=tri_to_patch,
            )
        else:
            # Component rendering only supported with shared hits
            if self.verbose:
                print("  WARNING: Component rendering requires use_shared_hits=True")
            return None

        if isinstance(result, ADCComponentResult):
            if self.verbose:
                print("\n" + "=" * 60)
                print("Component Rendering Complete")
                print(f"  Total paths: {result.total_paths:,}")
                print(f"  Components: ka, spm, directive, broad, coherent, incoherent, total")
                print("=" * 60)
            return result
        else:
            if self.verbose:
                print("  WARNING: BSDF doesn't support component extraction")
            return None

    def get_adc_numpy(self) -> np.ndarray:
        """
        Get ADC data as numpy array.

        Returns:
            ADC array of shape (n_tx, n_rx, n_samples, 2) with real/imag components
        """
        if self._last_result is None:
            raise RuntimeError("No render result available. Call render() first.")

        return self._last_result.adc_result.get_ri_array()

    def get_adc_complex(self) -> np.ndarray:
        """
        Get ADC data as complex numpy array.

        Returns:
            Complex ADC array of shape (n_tx, n_rx, n_samples)
        """
        if self._last_result is None:
            raise RuntimeError("No render result available. Call render() first.")

        return self._last_result.adc_result.get_complex()

    def get_reservoir_hits(self) -> Optional[ReservoirHits]:
        """
        Get reservoir hits from last render.

        Returns:
            ReservoirHits or None if no render has been performed
        """
        if self._last_result is None:
            return None
        return self._last_result.reservoir_hits

    @property
    def last_result(self) -> Optional[RenderResultRef]:
        """Get the last render result."""
        return self._last_result


__all__ = [
    'FMCWRendererRef',
    'RenderConfigRef',
    'RenderResultRef',
]
