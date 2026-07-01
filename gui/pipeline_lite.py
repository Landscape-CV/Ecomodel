"""
Lite-pipeline orchestrator for EcomodelMainWindow.

Wraps EcomodelLite in the same callback contract as run_ecomodel_pipeline so
EcomodelWorker can route to it transparently based on config.pipeline_type.

Pipeline steps (per tile):
  1. Load & normalise
  2. Ground removal (CSF)
  3. Intensity filter
  4. Leaf removal (RGI)
  5. Instance segmentation
  6. TreeQSM (cylinders)

Output layout (under the run_dir):
  {run_dir}/
      {tile_stem}/
          {tile_stem}_cylinders.txt      <- per-tile cylinder array
          {tile_stem}_leavesremoved.xyz  <- leaves-removed point cloud
          {tile_stem}_data.txt           <- mean + ground_z for the tile
      ecomodel_lite_cylinders.txt        <- all tiles concatenated (for ResultsPage)
      metadata.json                      <- written last; makes run visible to GUI
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from gui.config import EcomodelConfig

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class PipelineStopped(Exception):
    """Raised when the user clicks Stop."""


def _debug_save(tile_dir: Path, filename: str, data: np.ndarray) -> None:
    """Save an intermediate point cloud for debug inspection (silent on error)."""
    try:
        np.savetxt(str(tile_dir / filename), data)
    except Exception:
        pass


# Combined cylinder filename written to the run_dir root
_COMBINED_CYL_FILE = "ecomodel_lite_cylinders.txt"

# Steps executed per tile (for progress bar denominator)
_STEPS_PER_TILE = 6
_STEP_LABELS = [
    "Loading tile",
    "Ground removal (CSF)",
    "Intensity filter",
    "Leaf removal (RGI)",
    "Instance segmentation",
    "TreeQSM (cylinders)",
]


def run_ecomodel_lite_pipeline(
    config: EcomodelConfig,
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    tile_update_callback: Optional[Callable[[str, str, int, str], None]] = None,
) -> dict:
    """
    Execute the EcomodelLite pipeline for all LAS/LAZ tiles in config.input_folder.

    Parameters
    ----------
    config : EcomodelConfig
        Must have pipeline_type == "lite".  Only input_folder, results_folder,
        cylinder_filename, and lite_* fields are used.
    log_callback : callable(str)
        Receives free-form log lines for the GUI log panel.
    progress_callback : callable(int, int, str)
        Receives (current_step, total_steps, label) for the progress bar.
    should_stop : callable() -> bool
        Checked between steps; raises PipelineStopped when True.
    tile_update_callback : callable(str, str, int, str)
        Receives (tile_name, status, cyl_count, output_path) after each tile.

    Returns
    -------
    dict
        {"run_dir": Path, "cylinder_count": int, "tile_count": int}
    """
    from ecomodel_lite import EcomodelLite
    from gui.results_io import (make_run_dir, write_run_metadata,
                                save_point_cloud_snapshot, save_segment_labels_snapshot)
    from Utils.Utils import load_point_cloud

    def _log(msg: str) -> None:
        if log_callback:
            log_callback(msg)

    def _progress(step: int, total: int, label: str) -> None:
        if progress_callback:
            progress_callback(step, total, label)

    def _check_stop() -> None:
        if should_stop and should_stop():
            raise PipelineStopped("User requested stop.")

    def _tile_update(name: str, status: str, cyl_count: int, path: str) -> None:
        if tile_update_callback:
            tile_update_callback(name, status, cyl_count, path)

    # ── Discover tiles ────────────────────────────────────────────────────────
    input_path = Path(config.input_folder)
    tiles = sorted(
        list(input_path.glob("*.las")) + list(input_path.glob("*.laz"))
    )
    if not tiles:
        raise RuntimeError(
            f"No LAS/LAZ files found in {config.input_folder!r}. "
            "Check the input folder on the Configure page."
        )

    total_tiles = len(tiles)
    total_steps = total_tiles * _STEPS_PER_TILE + 1  # +1 for final concatenation
    _log(f"[Lite] Found {total_tiles} tile(s) in {input_path}\n")

    # ── Create run output directory ───────────────────────────────────────────
    run_name = getattr(config, "cylinder_filename", "ecomodel_lite")
    run_dir = make_run_dir(config.results_folder, run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    _log(f"[Lite] Output directory: {run_dir}\n")

    # ── Initialise EcomodelLite ───────────────────────────────────────────────
    model = EcomodelLite(
        results_folder=str(run_dir),
        intensity_threshold=config.lite_intensity_threshold,
        # CSF
        csf_cloth_resolution=config.lite_csf_cloth_resolution,
        csf_class_threshold=config.lite_csf_class_threshold,
        csf_iterations=config.lite_csf_iterations,
        csf_remove_underground=config.lite_csf_remove_underground,
        # Noise removal
        noise_voxel_size=config.lite_noise_voxel_size,
        noise_min_points=config.lite_noise_min_points,
        # RGI (shared with full pipeline fields)
        rgi_noise_percentile=config.rgi_noise_percentile,
        rgi_angle_deg=config.rgi_angle_deg,
        rgi_curv_thresh=config.rgi_curv_thresh,
        rgi_resid_thresh=config.rgi_resid_thresh,
        rgi_k=config.rgi_k,
        rgi_min_cluster_size=config.rgi_min_cluster_size,
        rgi_max_cluster_size=config.rgi_max_cluster_size,
        rgi_smooth_mode=config.rgi_smooth_mode,
        rgi_use_residual_test=config.rgi_use_residual_test,
        rgi_use_curvature_test=config.rgi_use_curvature_test,
        # QSM cover sets
        patch_diam1=config.lite_patch_diam1,
        ball_rad1=config.lite_ball_rad1,
        nmin1=config.lite_nmin1,
        patch_diam2_min=config.lite_patch_diam2_min,
        patch_diam2_max=config.lite_patch_diam2_max,
        ball_rad2=config.lite_ball_rad2,
        # Instance segmenter
        segmenter_type=config.lite_segmenter_type,
        treelearn_config_path=config.treelearn_config_path,
        treelearn_use_gpu=config.treelearn_use_gpu,
    )

    # ── Process tiles ─────────────────────────────────────────────────────────
    total_cylinders = 0
    total_points = 0
    processed = 0
    snap_blocks = []   # (world_xyz, labels) per tile -> queryable run snapshot
    label_offset = 0
    all_tree_metrics = []   # per-tree QSM attributes across all tiles
    tree_id_global = 0

    for i, tile_path in enumerate(tiles):
        _check_stop()

        tile_name = tile_path.stem
        tile_offset = i * _STEPS_PER_TILE

        _log(f"\n[Lite] ── Tile {i + 1}/{total_tiles}: {tile_name} ──\n")
        _tile_update(tile_name, "Running", -1, "")

        try:
            # ── Step 1: Load & normalise ──────────────────────────────────────
            _check_stop()
            _progress(tile_offset + 1, total_steps,
                      f"Tile {i+1}/{total_tiles} - {_STEP_LABELS[0]}")
            _log(f"[Lite]   [1/{_STEPS_PER_TILE}] Loading point cloud...\n")

            os.makedirs(str(run_dir / tile_name), exist_ok=True)
            _, full_data = load_point_cloud(
                str(tile_path), full_data=True,
                scalar_field=getattr(config, "scalar_field", "intensity"),
                normalize_scalar=getattr(config, "normalize_scalar", False),
            )
            full_data = model.normalize_point_cloud(full_data)
            _log(f"[Lite]   Loaded {len(full_data):,} points\n")

            # ── Step 2: Ground removal (CSF) ──────────────────────────────────
            _check_stop()
            _progress(tile_offset + 2, total_steps,
                      f"Tile {i+1}/{total_tiles} - {_STEP_LABELS[1]}")
            _log(f"[Lite]   [2/{_STEPS_PER_TILE}] Ground removal (CSF)...\n")

            full_data = model.remove_ground(full_data)
            if full_data is None:
                _log(f"[Lite]   WARNING: no non-ground points after CSF. Skipping tile.\n")
                _tile_update(tile_name, "Skipped", 0, "")
                continue
            _log(f"[Lite]   {len(full_data):,} non-ground points\n")
            if config.debug_mode:
                _debug_save(run_dir / tile_name, f"{tile_name}_debug_01_csf.xyz", full_data[:, :3])

            # ── Step 3: Intensity filter ──────────────────────────────────────
            _check_stop()
            _progress(tile_offset + 3, total_steps,
                      f"Tile {i+1}/{total_tiles} - {_STEP_LABELS[2]}")
            _log(f"[Lite]   [3/{_STEPS_PER_TILE}] Intensity filter "
                 f"(threshold={model.intensity_threshold})...\n")

            before = len(full_data)
            full_data = model.filter_intensity(full_data, model.intensity_threshold)
            _log(f"[Lite]   {before:,} to {len(full_data):,} points after intensity filter\n")
            if full_data.size < 100:
                _log(f"[Lite]   WARNING: too few points after intensity filter. Skipping tile.\n")
                _tile_update(tile_name, "Skipped", 0, "")
                continue
            if config.debug_mode:
                _debug_save(run_dir / tile_name, f"{tile_name}_debug_02_intensity.xyz", full_data[:, :3])

            # ── Step 4: Leaf removal (RGI) ────────────────────────────────────
            _check_stop()
            _progress(tile_offset + 4, total_steps,
                      f"Tile {i+1}/{total_tiles} - {_STEP_LABELS[3]}")
            if config.run_leaf_removal:
                _log(f"[Lite]   [4/{_STEPS_PER_TILE}] Leaf removal (RGI)...\n")
                full_data = model.remove_leaves_rgi(full_data)
                if full_data is None:
                    _log(f"[Lite]   WARNING: RGI leaf removal returned no wood. Skipping tile.\n")
                    _tile_update(tile_name, "Skipped", 0, "")
                    continue
                _log(f"[Lite]   RGI complete - {len(full_data):,} wood points retained\n")
                if config.debug_mode:
                    _debug_save(run_dir / tile_name, f"{tile_name}_debug_03_rgi_wood.xyz", full_data[:, :3])
            else:
                _log(f"[Lite]   [4/{_STEPS_PER_TILE}] Leaf removal skipped (run_leaf_removal off).\n")

            # ── Step 5: Instance segmentation ─────────────────────────────────
            _check_stop()
            _progress(tile_offset + 5, total_steps,
                      f"Tile {i+1}/{total_tiles} - {_STEP_LABELS[4]}")
            _log(f"[Lite]   [5/{_STEPS_PER_TILE}] Instance segmentation...\n")

            full_data, instance_labels = model.perform_instance_segmentation(
                full_data, output_dir=str(run_dir / tile_name)
            )
            if full_data is None or instance_labels is None:
                _log(f"[Lite]   WARNING: instance segmentation failed. Skipping tile.\n")
                _tile_update(tile_name, "Skipped", 0, "")
                continue
            n_trees = int(np.sum(np.unique(instance_labels) != -1))
            _log(f"[Lite]   Segmentation complete - {n_trees} tree segment(s)\n")
            if config.debug_mode:
                _debug_save(
                    run_dir / tile_name,
                    f"{tile_name}_debug_04_segmented.xyz",
                    np.concatenate([full_data[:, :3], instance_labels[:, np.newaxis]], axis=1),
                )

            # ── Step 6: TreeQSM ───────────────────────────────────────────────
            _check_stop()
            _progress(tile_offset + 6, total_steps,
                      f"Tile {i+1}/{total_tiles} - {_STEP_LABELS[5]}")
            qsm_method = getattr(config, "lite_qsm_method", "treeqsm")
            _log(f"[Lite]   [6/{_STEPS_PER_TILE}] QSM ({qsm_method})...\n")

            if qsm_method == "smartqsm":
                from gui.smartqsm_runner import run_smartqsm_on_segments
                cylinder_data = run_smartqsm_on_segments(
                    full_data, instance_labels, str(run_dir / tile_name), config, _log)
                tile_tree_metrics = []   # SmartQSM does not produce tree_data
            else:
                cylinder_data, tile_tree_metrics = model.get_cylinders(full_data, instance_labels)

            # ── Save tile results ─────────────────────────────────────────────
            cyl_file = run_dir / tile_name / f"{tile_name}_cylinders.txt"
            cylinder_out = model.unnormalize_point_cloud(cylinder_data.copy())
            np.savetxt(str(cyl_file), cylinder_out)

            unnorm_data = model.unnormalize_point_cloud(full_data.copy())
            with_labels = np.concatenate(
                (unnorm_data[:, :3], instance_labels[:, np.newaxis]), axis=1
            )
            np.savetxt(str(run_dir / tile_name / f"{tile_name}_leavesremoved.xyz"), with_labels)

            # Keep world-coord wood + labels for the queryable run snapshot.
            lab = instance_labels.copy()
            pos = lab >= 0
            lab[pos] += label_offset
            snap_blocks.append((unnorm_data[:, :3].copy(), lab))
            if pos.any():
                label_offset = int(lab[pos].max()) + 1

            # Per-tree metrics with a globally-unique tree id + originating tile.
            for tm in tile_tree_metrics:
                tm = dict(tm)
                tm["tree_id"] = tree_id_global
                tm["tile"] = tile_name
                tree_id_global += 1
                all_tree_metrics.append(tm)

            with open(str(run_dir / tile_name / f"{tile_name}_data.txt"), "w") as fh:
                fh.write(f"{model.mean[0]} {model.mean[1]} {model.mean[2]}\n")
                fh.write(str(model.ground_z))

        except PipelineStopped:
            raise
        except Exception as exc:
            _log(f"[Lite]   ERROR processing {tile_name}: {exc}\n")
            _tile_update(tile_name, "Error", 0, "")
            continue

        # Count cylinders from the saved file
        cyl_count = 0
        if cyl_file.exists() and cyl_file.stat().st_size > 0:
            try:
                data = np.loadtxt(cyl_file)
                if data.ndim == 1 and data.size > 0:
                    cyl_count = 1
                elif data.ndim == 2:
                    cyl_count = data.shape[0]
            except Exception:
                pass

        total_cylinders += cyl_count
        processed += 1
        _log(f"[Lite]   Done - {cyl_count} cylinder(s)\n")
        _tile_update(tile_name, "Done", cyl_count, str(cyl_file))

    # ── Concatenate all per-tile cylinder files ───────────────────────────────
    _progress(total_steps, total_steps, "Writing combined cylinder file")
    _log("[Lite] Concatenating cylinder files...\n")

    all_cyl_blocks = []
    for tile_path in tiles:
        tile_name = tile_path.stem
        cyl_file = run_dir / tile_name / f"{tile_name}_cylinders.txt"
        if cyl_file.exists() and cyl_file.stat().st_size > 0:
            try:
                data = np.loadtxt(cyl_file)
                if data.ndim == 1 and data.size > 0:
                    all_cyl_blocks.append(data.reshape(1, -1))
                elif data.ndim == 2 and data.shape[0] > 0:
                    all_cyl_blocks.append(data)
            except Exception:
                pass

    combined_cyl_path = run_dir / _COMBINED_CYL_FILE
    if all_cyl_blocks:
        combined = np.concatenate(all_cyl_blocks, axis=0)
        np.savetxt(str(combined_cyl_path), combined)
        total_cylinders = combined.shape[0]
        _log(f"[Lite] Combined cylinder file: {combined_cyl_path} ({total_cylinders} cylinders)\n")
    else:
        _log("[Lite] WARNING: no cylinders found across all tiles.\n")

    # ── Per-tree metrics table ────────────────────────────────────────────────
    if all_tree_metrics:
        from gui.results_io import save_tree_metrics
        tm_path = save_tree_metrics(run_dir, all_tree_metrics)
        _log(f"[Lite] Tree metrics: {tm_path} ({len(all_tree_metrics)} trees)\n")

    # ── Save the queryable snapshot (point_cloud.npy + segment_labels.npy) ────
    # Stored in normalised space (world - cloud_mean) to match the full pipeline;
    # the combined cylinder file is in world coords and the query reconciles them
    # via cloud_mean.  Without this, lite runs cannot be voxel/GPS-queried.
    cloud_mean_out = None
    if snap_blocks:
        world = np.concatenate([b[0] for b in snap_blocks], axis=0)
        labs = np.concatenate([b[1] for b in snap_blocks], axis=0)
        cloud_mean = world.mean(axis=0)
        norm = (world - cloud_mean).astype(np.float32)
        if len(norm) > 200000:
            sel = np.random.default_rng(0).choice(len(norm), 200000, replace=False)
            norm, labs = norm[sel], labs[sel]
        idx = save_point_cloud_snapshot(run_dir, norm)
        save_segment_labels_snapshot(run_dir, labs.astype(np.int32), indices=idx)
        cloud_mean_out = [float(v) for v in cloud_mean]
        total_points = int(len(world))
        _log(f"[Lite] Saved queryable snapshot ({total_points:,} pts).\n")

    # ── Write metadata.json so the Results page can discover this run ─────────
    original_cyl_name = getattr(config, "cylinder_filename", "ecomodel_lite")
    config.cylinder_filename = Path(_COMBINED_CYL_FILE).stem  # strip .txt

    write_run_metadata(
        run_dir=run_dir,
        config=config,
        cylinder_count=total_cylinders,
        point_count=total_points,
        cloud_mean=cloud_mean_out,
    )
    config.cylinder_filename = original_cyl_name  # restore

    _log(
        f"\n[Lite] Pipeline complete. "
        f"{processed}/{total_tiles} tile(s) processed, "
        f"{total_cylinders} total cylinder(s).\n"
    )

    return {
        "run_dir": run_dir,
        "cylinder_count": total_cylinders,
        "tile_count": processed,
    }
