from __future__ import annotations
from dataclasses import dataclass


@dataclass
class EcomodelConfig:
    # ── I/O ──────────────────────────────────────────────────────────────────
    input_folder: str = ""
    results_folder: str = "results"
    cylinder_filename: str = "ecomodel_cylinder_data"

    # ── Scalar field mapping ──────────────────────────────────────────────────
    # Which LAS per-point dimension fills the working "intensity" column.
    # Default "intensity"; set to e.g. "Reflectance" for RIEGL-style exports
    # whose standard intensity field is empty.  normalize_scalar rescales the
    # chosen field to 0-65535 so negative/fractional units work with the
    # positive intensity thresholds.
    scalar_field: str = "intensity"
    normalize_scalar: bool = False

    # ── Tile subdivision ─────────────────────────────────────────────────────
    cube_size: float = 10.0
    meter_conversion: float = 1.0

    # ── Ground filtering (CSF) ───────────────────────────────────────────────
    csf_band_size: float = 0.1
    csf_threshold: int = 20
    csf_offset: float = 0.2
    remove_under_ground: bool = True

    # ── Terrain model ────────────────────────────────────────────────────────
    terrain_grid_size: float = 0.1

    # ── Optional cleanup / denoise ───────────────────────────────────────────
    use_denoise: bool = False
    denoise_grid_size: float = 0.1
    denoise_min_points: int = 10
    denoise_resolution: float = 0.05

    # ── Duplicate removal ────────────────────────────────────────────────────
    remove_duplicates: bool = False

    # ── Tree segmentation ────────────────────────────────────────────────────
    segment_intensity_threshold: int = 0
    save_clusters: bool = False

    # ── QSM ──────────────────────────────────────────────────────────────────
    # 0 = no intensity pre-filter (RGI handles leaf removal). A non-zero value
    # discards wood on reflectance-valued scans where wood is not high-intensity.
    qsm_intensity_threshold: int = 0
    save_leaf_removal_output: bool = False
    run_qsm: bool = True
    # EXPERIMENT (2026-07-10): default off so SmartQSM gets the full leaf-on cloud
    # (matches Modal's leaf-on config; avoids RGI stripping thin-branch wood ->
    # fewer gaps). Flip back to True to re-enable RGI leaf removal.
    run_leaf_removal: bool = False

    # ── Visualisation ────────────────────────────────────────────────────────
    create_cylinder_plot: bool = False

    # ── Cover set placeholders ───────────────────────────────────────────────
    # Stored here for GUI/config completeness. These are not yet fully wired
    # through all Ecomodel methods unless explicitly passed downstream.
    patch_diam1: float = 0.15
    ball_rad1: float = 0.15
    nmin1: int = 25
    patch_diam2_min: float = 0.05
    patch_diam2_max: float = 0.08
    ball_rad2: float = 0.09

    # ── RGI placeholders ─────────────────────────────────────────────────────
    # Stored here for GUI/config completeness. These are not yet fully wired
    # into classify_wood_leaf() unless explicitly forwarded by the pipeline.
    rgi_noise_percentile: int = 0
    rgi_angle_deg: float = 7.0
    rgi_curv_thresh: float = 0.07
    rgi_resid_thresh: float = 0.05
    rgi_k: int = 100
    rgi_min_cluster_size: int = 40
    rgi_max_cluster_size: int = 100000
    rgi_smooth_mode: bool = True
    rgi_use_residual_test: bool = True
    rgi_use_curvature_test: bool = True

    # ── Checkpointing ────────────────────────────────────────────────────────
    checkpoint_name: str = "ecomodel_checkpoint"
    checkpoint_folder: str = "pickle"
    use_checkpoint: bool = False
    save_checkpoint: bool = False
    checkpoint_resume_file: str = ""   # full path to .pickle file to resume from

    # ── Pipeline mode ─────────────────────────────────────────────────────────
    pipeline_type: str = "full"        # "full" | "lite"

    # ── Debug mode ────────────────────────────────────────────────────────────
    debug_mode: bool = False  # save intermediates at each step + auto-report

    # ── Lite: general ─────────────────────────────────────────────────────────
    lite_intensity_threshold: int = 0

    # ── Lite: CSF ground removal (different param space from full-pipeline CSF) ──
    lite_csf_cloth_resolution: float = 2.0
    lite_csf_class_threshold: float = 0.5
    lite_csf_iterations: int = 500
    lite_csf_remove_underground: bool = True

    # ── Lite: noise removal ───────────────────────────────────────────────────
    lite_noise_voxel_size: float = 0.25
    lite_noise_min_points: int = 100

    # ── Lite: QSM cover sets (different defaults from full-pipeline cover sets) ─
    lite_patch_diam1: float = 0.025
    lite_ball_rad1: float = 0.03
    lite_nmin1: int = 5
    lite_patch_diam2_min: float = 0.05
    lite_patch_diam2_max: float = 0.08
    lite_ball_rad2: float = 0.09
    # RGI params are shared with the full pipeline (same library, same defaults)

    # ── Lite: instance segmenter ──────────────────────────────────────────────
    lite_segmenter_type: str = "scanline"   # "scanline" | "treelearn"
    treelearn_config_path: str = ""         # path to TreeLearn YAML config
    treelearn_use_gpu: bool = True          # use CUDA; False = CPU (very slow)
    # Single-tree mode: skip segmentation and treat the whole tile as ONE tree,
    # so the QSM step reconstructs it in one piece (avoids over-segmenting a
    # single tree into fragments). Use ONLY when the input is a single tree; a
    # multi-tree tile would collapse into one tangled trunk.
    lite_single_tree: bool = False

    # ── Lite: QSM method ──────────────────────────────────────────────────────
    lite_qsm_method: str = "treeqsm"        # "treeqsm" | "smartqsm"
    smartqsm_dir: str = ""
    smartqsm_python: str = ""
    smartqsm_config: str = ""
    # Remote GPU service URL. When set (or overridden by env SMARTQSM_URL),
    # segments are POSTed to this endpoint instead of run via local subprocess.
    # Defaults to the lab's deployed Modal serverless GPU backend, so the app
    # works out of the box with no setup; env SMARTQSM_URL still overrides.
    smartqsm_url: str = "https://nisulay--smartqsm-web.modal.run"