"""
EcomodelMainWindow - PySide6 main window for the Ecomodel pipeline.

Layout
──────
  Left panel  (scrollable QScrollArea)
      Parameter inputs grouped by pipeline stage:
        Input / Output
        Tile Subdivision
        Ground Filtering (CSF)
        Terrain Model (DEM)
        Optional Cleanup / Denoise
        Tree Segmentation
        QSM
        Cover Sets        (stored in config; not yet wired into ecomodel.py)
        RGI Leaf/Wood     (stored in config; not yet wired into ecomodel.py)
        Checkpoint

  Right panel
        Run / Stop buttons
        Progress bar
        Step label
        Log text area (read-only, monospaced)
        Clear Log button

Threading model:
    QThread + EcomodelWorker(QObject).moveToThread(thread)
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.config import EcomodelConfig
from gui.query_widgets import QueryPage
from gui.results_widgets import ResultsPage
from gui.worker import EcomodelWorker


class EcomodelMainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ecomodel")
        self.resize(1150, 820)

        self._thread: QThread | None = None
        self._worker: EcomodelWorker | None = None
        self._defaults = EcomodelConfig()
        self._last_run_dir: "Path | None" = None   # set after each run for report button

        # ── Stacked widget: page 0 = pipeline, page 1 = results, page 2 = query
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        # Page 0 - pipeline
        pipeline_page = QWidget()
        pipeline_root = QHBoxLayout(pipeline_page)
        splitter = QSplitter(Qt.Horizontal)
        pipeline_root.addWidget(splitter)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        param_widget = QWidget()
        scroll.setWidget(param_widget)
        splitter.addWidget(scroll)

        right = QWidget()
        splitter.addWidget(right)
        splitter.setSizes([560, 590])

        self._build_param_panel(param_widget)
        self._build_right_panel(right)

        self._stack.addWidget(pipeline_page)          # index 0

        # Initialise status bar before connecting signals that write to it.
        self.setStatusBar(QStatusBar())

        # Page 1 - results viewer
        self._results_page = ResultsPage(
            results_folder=self._defaults.results_folder,
        )
        self._results_page.back_requested.connect(
            lambda: self._stack.setCurrentIndex(0)
        )
        self._results_page.status_message.connect(
            lambda msg: self.statusBar().showMessage(msg)
        )
        self._results_page.query_requested.connect(self._show_query_page)
        self._stack.addWidget(self._results_page)     # index 1

        # Page 2 - query mode
        self._query_page = QueryPage(results_folder=self._defaults.results_folder)
        self._query_page.back_requested.connect(lambda: self._stack.setCurrentIndex(1))
        self._query_page.status_message.connect(
            lambda msg: self.statusBar().showMessage(msg)
        )
        self._stack.addWidget(self._query_page)       # index 2

    # ── Parameter panel ───────────────────────────────────────────────────────

    def _build_param_panel(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)
        layout.setAlignment(Qt.AlignTop)

        # ── Pipeline mode selector ────────────────────────────────────────────
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Pipeline Mode:"))
        self._pipeline_mode = QComboBox()
        self._pipeline_mode.addItems(["Full (ecomodel)", "Lite (ecomodel_lite)"])
        self._pipeline_mode.setToolTip(
            "Full: 14-step GPU-accelerated pipeline for large scans.\n"
            "Lite: 5-step CPU pipeline - ground removal, leaf removal, "
            "instance segmentation, TreeQSM."
        )
        self._pipeline_mode.currentIndexChanged.connect(self._on_pipeline_mode_changed)
        mode_row.addWidget(self._pipeline_mode, stretch=1)
        layout.addLayout(mode_row)

        # ── I/O ──────────────────────────────────────────────────────────────
        io = QGroupBox("Input / Output")
        g = QGridLayout(io)

        g.addWidget(QLabel("LAS/LAZ Folder:"), 0, 0)
        self._input_folder = QLineEdit()
        self._input_folder.setPlaceholderText("Folder containing .las / .laz tiles")
        self._input_folder.editingFinished.connect(
            lambda: self._auto_populate_names(self._input_folder.text().strip())
            if self._input_folder.text().strip() else None
        )
        g.addWidget(self._input_folder, 0, 1)
        b_in = QPushButton("Browse…")
        b_in.clicked.connect(self._browse_input)
        g.addWidget(b_in, 0, 2)

        g.addWidget(QLabel("Results Folder:"), 1, 0)
        self._results_folder = QLineEdit(self._defaults.results_folder)
        g.addWidget(self._results_folder, 1, 1)
        b_out = QPushButton("Browse…")
        b_out.clicked.connect(self._browse_results)
        g.addWidget(b_out, 1, 2)

        g.addWidget(QLabel("Cylinder Filename:"), 2, 0)
        self._cylinder_filename = QLineEdit(self._defaults.cylinder_filename)
        g.addWidget(self._cylinder_filename, 2, 1, 1, 2)

        g.addWidget(QLabel("Scalar Field:"), 3, 0)
        self._scalar_field = QComboBox()
        self._scalar_field.setEditable(True)
        self._scalar_field.addItem(self._defaults.scalar_field)
        self._scalar_field.setToolTip(
            "LAS per-point field used as the intensity channel.\n"
            "Defaults to 'intensity'. For RIEGL-style scans whose intensity is "
            "empty, pick 'Reflectance'. The list is populated from the input file."
        )
        self._scalar_field.currentTextChanged.connect(self._on_scalar_field_changed)
        g.addWidget(self._scalar_field, 3, 1, 1, 2)

        self._normalize_scalar = QCheckBox("Normalize scalar field to 0-65535")
        self._normalize_scalar.setChecked(self._defaults.normalize_scalar)
        self._normalize_scalar.setToolTip(
            "Rescale the chosen field to the 0-65535 range so fields with "
            "negative or fractional units (e.g. reflectance in dB) work with "
            "the positive intensity thresholds. Auto-enabled for non-intensity "
            "fields; rescaling is per-file."
        )
        g.addWidget(self._normalize_scalar, 4, 0, 1, 3)

        layout.addWidget(io)

        # ── Lite: General (shown only in Lite mode) ──────────────────────────
        lg = QGroupBox("Lite Settings")
        ll = QGridLayout(lg)

        ll.addWidget(QLabel("Intensity Threshold:"), 0, 0)
        self._lite_intensity = self._spin(0, 200000, self._defaults.lite_intensity_threshold)
        self._lite_intensity.setToolTip("Remove points below this intensity before leaf removal.")
        ll.addWidget(self._lite_intensity, 0, 1)

        ll.addWidget(QLabel("Segmenter:"), 1, 0)
        self._lite_segmenter = QComboBox()
        self._lite_segmenter.addItems(["Scanline (default)", "TreeLearn (neural net)"])
        self._lite_segmenter.setToolTip(
            "Scanline: fast geometric segmenter.\n"
            "TreeLearn: deep-learning segmenter - handles overlapping canopy better "
            "but requires a GPU and pre-downloaded model weights."
        )
        ll.addWidget(self._lite_segmenter, 1, 1)

        # ── TreeLearn sub-controls (hidden when Scanline selected) ────────────
        self._treelearn_widget = QWidget()
        tlw = QGridLayout(self._treelearn_widget)
        tlw.setContentsMargins(0, 0, 0, 0)

        tlw.addWidget(QLabel("Config YAML:"), 0, 0)
        self._treelearn_config = QLineEdit(self._defaults.treelearn_config_path)
        self._treelearn_config.setPlaceholderText("Path to TreeLearn pipeline YAML...")
        self._treelearn_config.setToolTip(
            "YAML config file for TreeLearn (the 'pretrain' key inside must point "
            "to a downloaded .pth weights file)."
        )
        tlw.addWidget(self._treelearn_config, 0, 1)
        _tl_browse = QPushButton("Browse…")
        _tl_browse.clicked.connect(self._browse_treelearn_config)
        tlw.addWidget(_tl_browse, 0, 2)

        self._treelearn_gpu = QCheckBox("Use GPU (CUDA)")
        self._treelearn_gpu.setChecked(self._defaults.treelearn_use_gpu)
        self._treelearn_gpu.setToolTip(
            "Run TreeLearn on the GPU (recommended).\n"
            "Uncheck to use CPU - very slow, for testing only."
        )
        tlw.addWidget(self._treelearn_gpu, 1, 0, 1, 3)

        self._treelearn_widget.setVisible(False)
        ll.addWidget(self._treelearn_widget, 2, 0, 1, 2)

        self._lite_segmenter.currentIndexChanged.connect(self._on_segmenter_changed)

        ll.addWidget(QLabel("QSM Method:"), 3, 0)
        self._lite_qsm_method = QComboBox()
        self._lite_qsm_method.addItems(["TreeQSM", "SmartQSM"])
        self._lite_qsm_method.setToolTip("SmartQSM is an external tool installed separately.")
        ll.addWidget(self._lite_qsm_method, 3, 1)

        self._smartqsm_widget = QWidget()
        sqw = QGridLayout(self._smartqsm_widget)
        sqw.setContentsMargins(0, 0, 0, 0)
        self._smartqsm_dir = QLineEdit(self._defaults.smartqsm_dir)
        self._smartqsm_dir.setPlaceholderText("SmartQSM checkout folder...")
        _sq_db = QPushButton("Browse…")
        _sq_db.clicked.connect(self._browse_smartqsm_dir)
        sqw.addWidget(QLabel("SmartQSM dir:"), 0, 0)
        sqw.addWidget(self._smartqsm_dir, 0, 1)
        sqw.addWidget(_sq_db, 0, 2)
        self._smartqsm_python = QLineEdit(self._defaults.smartqsm_python)
        self._smartqsm_python.setPlaceholderText("SmartQSM venv python.exe...")
        _sq_pb = QPushButton("Browse…")
        _sq_pb.clicked.connect(self._browse_smartqsm_python)
        sqw.addWidget(QLabel("SmartQSM python:"), 1, 0)
        sqw.addWidget(self._smartqsm_python, 1, 1)
        sqw.addWidget(_sq_pb, 1, 2)
        self._smartqsm_config = QLineEdit(self._defaults.smartqsm_config)
        self._smartqsm_config.setPlaceholderText("SmartQSM config YAML...")
        _sq_cb = QPushButton("Browse…")
        _sq_cb.clicked.connect(self._browse_smartqsm_config)
        sqw.addWidget(QLabel("SmartQSM config:"), 2, 0)
        sqw.addWidget(self._smartqsm_config, 2, 1)
        sqw.addWidget(_sq_cb, 2, 2)
        self._smartqsm_widget.setVisible(False)
        ll.addWidget(self._smartqsm_widget, 4, 0, 1, 2)
        self._lite_qsm_method.currentIndexChanged.connect(self._on_lite_qsm_method_changed)

        lg.setVisible(False)
        layout.addWidget(lg)

        # ── Lite: Ground Removal (CSF) ────────────────────────────────────────
        lcg = QGroupBox("Ground Removal (CSF)")
        lcg.setCheckable(True)
        lcg.setChecked(False)
        lcg.setToolTip("Check the title to enable editing of this section")
        lcl = QGridLayout(lcg)
        lcl.addWidget(QLabel("Cloth Resolution:"), 0, 0)
        self._lite_csf_cloth = self._dspin(0.1, 20.0, self._defaults.lite_csf_cloth_resolution, 2)
        self._lite_csf_cloth.setToolTip("Larger = coarser ground surface. Typical: 0.5-5.")
        lcl.addWidget(self._lite_csf_cloth, 0, 1)
        lcl.addWidget(QLabel("Class Threshold:"), 1, 0)
        self._lite_csf_thresh = self._dspin(0.01, 5.0, self._defaults.lite_csf_class_threshold, 3)
        self._lite_csf_thresh.setToolTip("Points within this distance of cloth = ground.")
        lcl.addWidget(self._lite_csf_thresh, 1, 1)
        lcl.addWidget(QLabel("Iterations:"), 2, 0)
        self._lite_csf_iters = self._spin(50, 5000, self._defaults.lite_csf_iterations)
        lcl.addWidget(self._lite_csf_iters, 2, 1)
        self._lite_csf_underground = QCheckBox("Remove Underground Points")
        self._lite_csf_underground.setChecked(self._defaults.lite_csf_remove_underground)
        lcl.addWidget(self._lite_csf_underground, 3, 0, 1, 2)
        lcg.setVisible(False)
        layout.addWidget(lcg)

        # ── Lite: Noise Removal ───────────────────────────────────────────────
        lng = QGroupBox("Noise Removal")
        lng.setCheckable(True)
        lng.setChecked(False)
        lng.setToolTip("Check the title to enable editing of this section")
        lnl = QGridLayout(lng)
        lnl.addWidget(QLabel("Voxel Size (m):"), 0, 0)
        self._lite_noise_voxel = self._dspin(0.01, 5.0, self._defaults.lite_noise_voxel_size, 3)
        self._lite_noise_voxel.setToolTip("Voxel grid size for cluster separation.")
        lnl.addWidget(self._lite_noise_voxel, 0, 1)
        lnl.addWidget(QLabel("Min Points:"), 1, 0)
        self._lite_noise_minpts = self._spin(1, 10000, self._defaults.lite_noise_min_points)
        self._lite_noise_minpts.setToolTip("Clusters smaller than this are removed as noise.")
        lnl.addWidget(self._lite_noise_minpts, 1, 1)
        lng.setVisible(False)
        layout.addWidget(lng)

        # ── Lite: QSM Cover Sets ──────────────────────────────────────────────
        lqg = QGroupBox("QSM Cover Sets")
        lqg.setCheckable(True)
        lqg.setChecked(False)
        lqg.setToolTip("Check the title to enable editing of this section")
        lql = QGridLayout(lqg)
        lite_cs_fields = [
            ("Patch Diam 1:",     "_lite_patch_diam1",     0.001, 2.0, self._defaults.lite_patch_diam1,     4),
            ("Ball Rad 1:",       "_lite_ball_rad1",       0.001, 2.0, self._defaults.lite_ball_rad1,       4),
            ("Patch Diam 2 Min:", "_lite_patch_diam2_min", 0.001, 2.0, self._defaults.lite_patch_diam2_min, 4),
            ("Patch Diam 2 Max:", "_lite_patch_diam2_max", 0.001, 2.0, self._defaults.lite_patch_diam2_max, 4),
            ("Ball Rad 2:",       "_lite_ball_rad2",       0.001, 2.0, self._defaults.lite_ball_rad2,       4),
        ]
        for r, (lbl, attr, lo, hi, val, dec) in enumerate(lite_cs_fields):
            lql.addWidget(QLabel(lbl), r, 0)
            w = self._dspin(lo, hi, val, dec)
            setattr(self, attr, w)
            lql.addWidget(w, r, 1)
        n = len(lite_cs_fields)
        lql.addWidget(QLabel("Nmin 1:"), n, 0)
        self._lite_nmin1 = self._spin(1, 500, self._defaults.lite_nmin1)
        self._lite_nmin1.setToolTip("Minimum points per cover-set ball (lower = more sets on sparse data).")
        lql.addWidget(self._lite_nmin1, n, 1)
        lqg.setVisible(False)
        layout.addWidget(lqg)

        # Collect all lite-only groups for show/hide toggling
        self._lite_groups = [lg, lcg, lng, lqg]
        self._lite_cover_group = lqg   # TreeQSM-only; hidden when SmartQSM selected

        # ── Tile subdivision ──────────────────────────────────────────────────
        tg = QGroupBox("Tile Subdivision")
        tg.setCheckable(True)
        tg.setChecked(False)
        tg.setToolTip("Check the title to enable editing of this section")
        tl = QGridLayout(tg)

        tl.addWidget(QLabel("Cube Size (m):"), 0, 0)
        self._cube_size = self._dspin(0.5, 500.0, self._defaults.cube_size, 1)
        tl.addWidget(self._cube_size, 0, 1)

        tl.addWidget(QLabel("Metre Conversion:"), 1, 0)
        self._meter_conversion = self._dspin(0.001, 1000.0, self._defaults.meter_conversion, 4)
        tl.addWidget(self._meter_conversion, 1, 1)

        reset_sub_btn = QPushButton("Reset to Defaults")
        reset_sub_btn.clicked.connect(self._reset_subdivision_defaults)
        tl.addWidget(reset_sub_btn, 2, 0, 1, 2)

        layout.addWidget(tg)

        # ── Ground filtering (CSF) ────────────────────────────────────────────
        cg = QGroupBox("Ground Filtering (CSF)")
        cg.setCheckable(True)
        cg.setChecked(False)
        cg.setToolTip("Check the title to enable editing of this section")
        cl = QGridLayout(cg)

        cl.addWidget(QLabel("Band Size:"), 0, 0)
        self._csf_band_size = self._dspin(0.01, 10.0, self._defaults.csf_band_size, 3)
        cl.addWidget(self._csf_band_size, 0, 1)

        cl.addWidget(QLabel("Threshold:"), 1, 0)
        self._csf_threshold = self._spin(1, 1000, self._defaults.csf_threshold)
        cl.addWidget(self._csf_threshold, 1, 1)

        cl.addWidget(QLabel("Offset:"), 2, 0)
        self._csf_offset = self._dspin(0.0, 10.0, self._defaults.csf_offset, 3)
        cl.addWidget(self._csf_offset, 2, 1)

        self._remove_underground = QCheckBox("Remove Underground Points")
        self._remove_underground.setChecked(self._defaults.remove_under_ground)
        cl.addWidget(self._remove_underground, 3, 0, 1, 2)

        reset_csf_btn = QPushButton("Reset to Defaults")
        reset_csf_btn.clicked.connect(self._reset_csf_defaults)
        cl.addWidget(reset_csf_btn, 4, 0, 1, 2)

        layout.addWidget(cg)

        # ── Terrain model ─────────────────────────────────────────────────────
        tmg = QGroupBox("Terrain Model (DEM)")
        tmg.setCheckable(True)
        tmg.setChecked(False)
        tmg.setToolTip("Check the title to enable editing of this section")
        tml = QGridLayout(tmg)

        tml.addWidget(QLabel("Grid Size (m):"), 0, 0)
        self._terrain_grid_size = self._dspin(0.01, 10.0, self._defaults.terrain_grid_size, 3)
        tml.addWidget(self._terrain_grid_size, 0, 1)

        reset_terrain_btn = QPushButton("Reset to Defaults")
        reset_terrain_btn.clicked.connect(self._reset_terrain_defaults)
        tml.addWidget(reset_terrain_btn, 1, 0, 1, 2)

        layout.addWidget(tmg)

        # ── Optional cleanup / denoise ────────────────────────────────────────
        og = QGroupBox("Optional Cleanup / Denoise")
        og.setCheckable(True)
        og.setChecked(False)
        og.setToolTip("Check the title to enable editing of this section")
        ol = QGridLayout(og)

        self._remove_duplicates = QCheckBox("Remove Duplicate Points")
        self._remove_duplicates.setChecked(self._defaults.remove_duplicates)
        ol.addWidget(self._remove_duplicates, 0, 0, 1, 2)

        self._use_denoise = QCheckBox("Run Denoise After Subdivision")
        self._use_denoise.setChecked(self._defaults.use_denoise)
        ol.addWidget(self._use_denoise, 1, 0, 1, 2)

        ol.addWidget(QLabel("Denoise Grid Size:"), 2, 0)
        self._denoise_grid_size = self._dspin(0.01, 10.0, self._defaults.denoise_grid_size, 3)
        ol.addWidget(self._denoise_grid_size, 2, 1)

        ol.addWidget(QLabel("Denoise Min Points:"), 3, 0)
        self._denoise_min_points = self._spin(1, 100000, self._defaults.denoise_min_points)
        ol.addWidget(self._denoise_min_points, 3, 1)

        ol.addWidget(QLabel("Denoise Resolution:"), 4, 0)
        self._denoise_resolution = self._dspin(0.001, 5.0, self._defaults.denoise_resolution, 3)
        ol.addWidget(self._denoise_resolution, 4, 1)

        reset_cleanup_btn = QPushButton("Reset to Defaults")
        reset_cleanup_btn.clicked.connect(self._reset_cleanup_defaults)
        ol.addWidget(reset_cleanup_btn, 5, 0, 1, 2)

        layout.addWidget(og)

        # ── Tree segmentation ─────────────────────────────────────────────────
        sg = QGroupBox("Tree Segmentation")
        sg.setCheckable(True)
        sg.setChecked(False)
        sg.setToolTip("Check the title to enable editing of this section")
        sl = QGridLayout(sg)

        sl.addWidget(QLabel("Intensity Threshold:"), 0, 0)
        self._seg_intensity = self._spin(0, 100000, self._defaults.segment_intensity_threshold)
        sl.addWidget(self._seg_intensity, 0, 1)

        self._save_clusters = QCheckBox("Save Intermediate Clusters")
        self._save_clusters.setChecked(self._defaults.save_clusters)
        sl.addWidget(self._save_clusters, 1, 0, 1, 2)

        reset_seg_btn = QPushButton("Reset to Defaults")
        reset_seg_btn.clicked.connect(self._reset_segmentation_defaults)
        sl.addWidget(reset_seg_btn, 2, 0, 1, 2)

        layout.addWidget(sg)

        # ── QSM ──────────────────────────────────────────────────────────────
        qg = QGroupBox("QSM / Outputs")
        qg.setCheckable(True)
        qg.setChecked(False)
        qg.setToolTip("Check the title to enable editing of this section")
        ql = QGridLayout(qg)

        self._run_qsm = QCheckBox("Run QSM + RGI")
        self._run_qsm.setChecked(self._defaults.run_qsm)
        ql.addWidget(self._run_qsm, 0, 0, 1, 2)

        self._run_leaf_removal = QCheckBox("Run Leaf Removal")
        self._run_leaf_removal.setChecked(getattr(self._defaults, "run_leaf_removal", True))
        ql.addWidget(self._run_leaf_removal, 1, 0, 1, 2)

        ql.addWidget(QLabel("Intensity Threshold:"), 2, 0)
        self._qsm_intensity = self._spin(0, 200000, self._defaults.qsm_intensity_threshold)
        ql.addWidget(self._qsm_intensity, 2, 1)

        self._save_leaf_output = QCheckBox("Save Leaf Removal Output")
        self._save_leaf_output.setChecked(self._defaults.save_leaf_removal_output)
        ql.addWidget(self._save_leaf_output, 3, 0, 1, 2)

        self._create_cylinder_plot = QCheckBox("Create Cylinder Plot")
        self._create_cylinder_plot.setChecked(self._defaults.create_cylinder_plot)
        ql.addWidget(self._create_cylinder_plot, 4, 0, 1, 2)

        reset_qsm_btn = QPushButton("Reset to Defaults")
        reset_qsm_btn.clicked.connect(self._reset_qsm_defaults)
        ql.addWidget(reset_qsm_btn, 5, 0, 1, 2)

        layout.addWidget(qg)

        # ── Cover sets ───────────────────────────────────────────────────────────
        csg = QGroupBox("Cover Sets (not active in Full pipeline)")
        csg.setCheckable(True)
        csg.setChecked(False)
        csg.setEnabled(False)
        csg.setToolTip("These cover-set parameters are not yet wired into the full pipeline.")
        csl = QGridLayout(csg)

        cs_fields = [
            ("Patch Diam 1:", "_patch_diam1", 0.001, 5.0, self._defaults.patch_diam1, 4),
            ("Ball Rad 1:", "_ball_rad1", 0.001, 5.0, self._defaults.ball_rad1, 4),
            ("Patch Diam 2 Min:", "_patch_diam2_min", 0.001, 5.0, self._defaults.patch_diam2_min, 4),
            ("Patch Diam 2 Max:", "_patch_diam2_max", 0.001, 5.0, self._defaults.patch_diam2_max, 4),
            ("Ball Rad 2:", "_ball_rad2", 0.001, 5.0, self._defaults.ball_rad2, 4),
        ]
        for row, (lbl, attr, lo, hi, val, dec) in enumerate(cs_fields):
            csl.addWidget(QLabel(lbl), row, 0)
            w = self._dspin(lo, hi, val, dec)
            setattr(self, attr, w)
            csl.addWidget(w, row, 1)

        n = len(cs_fields)
        csl.addWidget(QLabel("Nmin 1:"), n, 0)
        self._nmin1 = self._spin(1, 1000, self._defaults.nmin1)
        csl.addWidget(self._nmin1, n, 1)

        reset_cover_btn = QPushButton("Reset to Defaults")
        reset_cover_btn.clicked.connect(self._reset_cover_defaults)
        csl.addWidget(reset_cover_btn, n + 1, 0, 1, 2)

        layout.addWidget(csg)

        # ── RGI leaf/wood separation ──────────────────────────────────────────
        rg = QGroupBox("RGI Leaf/Wood Separation")
        self._rgi_group = rg   # keep reference for show/hide in lite mode
        rg.setCheckable(True)
        rg.setChecked(False)
        rg.setToolTip("Check the title to enable editing of this section")
        rl = QGridLayout(rg)

        rgi_fields = [
            ("Noise Percentile:", "_rgi_noise_percentile", "spin", 0, 100, self._defaults.rgi_noise_percentile, 0),
            ("Angle (deg):", "_rgi_angle_deg", "dspin", 0.1, 90.0, self._defaults.rgi_angle_deg, 2),
            ("Curvature Thresh:", "_rgi_curv_thresh", "dspin", 1e-4, 1.0, self._defaults.rgi_curv_thresh, 4),
            ("Residual Thresh:", "_rgi_resid_thresh", "dspin", 1e-4, 1.0, self._defaults.rgi_resid_thresh, 4),
            ("K (neighbours):", "_rgi_k", "spin", 5, 1000, self._defaults.rgi_k, 0),
            ("Min Cluster Size:", "_rgi_min_cluster_size", "spin", 1, 100000, self._defaults.rgi_min_cluster_size, 0),
            ("Max Cluster Size:", "_rgi_max_cluster_size", "spin", 1, 10000000, self._defaults.rgi_max_cluster_size, 0),
        ]
        for row, (lbl, attr, kind, lo, hi, val, dec) in enumerate(rgi_fields):
            rl.addWidget(QLabel(lbl), row, 0)
            if kind == "spin":
                w = self._spin(int(lo), int(hi), int(val))
            else:
                w = self._dspin(lo, hi, val, dec)
            setattr(self, attr, w)
            rl.addWidget(w, row, 1)

        off = len(rgi_fields)
        self._rgi_smooth_mode = QCheckBox("Smooth Mode")
        self._rgi_smooth_mode.setChecked(self._defaults.rgi_smooth_mode)
        rl.addWidget(self._rgi_smooth_mode, off, 0, 1, 2)

        self._rgi_use_residual = QCheckBox("Use Residual Test")
        self._rgi_use_residual.setChecked(self._defaults.rgi_use_residual_test)
        rl.addWidget(self._rgi_use_residual, off + 1, 0, 1, 2)

        self._rgi_use_curvature = QCheckBox("Use Curvature Test")
        self._rgi_use_curvature.setChecked(self._defaults.rgi_use_curvature_test)
        rl.addWidget(self._rgi_use_curvature, off + 2, 0, 1, 2)

        reset_rgi_btn = QPushButton("Reset to Defaults")
        reset_rgi_btn.clicked.connect(self._reset_rgi_defaults)
        rl.addWidget(reset_rgi_btn, off + 3, 0, 1, 2)

        layout.addWidget(rg)

        # ── Checkpoint ────────────────────────────────────────────────────────
        pkg = QGroupBox("Checkpoint")
        pkg.setCheckable(True)
        pkg.setChecked(False)
        pkg.setToolTip("Check the title to enable editing of this section")
        pkl = QGridLayout(pkg)

        pkl.addWidget(QLabel("Name:"), 0, 0)
        self._checkpoint_name = QLineEdit(self._defaults.checkpoint_name)
        pkl.addWidget(self._checkpoint_name, 0, 1)

        pkl.addWidget(QLabel("Folder:"), 1, 0)
        self._checkpoint_folder = QLineEdit(self._defaults.checkpoint_folder)
        pkl.addWidget(self._checkpoint_folder, 1, 1)
        browse_folder_btn = QPushButton("…")
        browse_folder_btn.setMaximumWidth(30)
        browse_folder_btn.clicked.connect(self._browse_checkpoint_folder)
        pkl.addWidget(browse_folder_btn, 1, 2)

        self._save_checkpoint = QCheckBox("Save Checkpoints During Run")
        self._save_checkpoint.setChecked(getattr(self._defaults, "save_checkpoint", False))
        pkl.addWidget(self._save_checkpoint, 2, 0, 1, 3)

        self._use_checkpoint = QCheckBox("Resume from Checkpoint")
        self._use_checkpoint.setChecked(self._defaults.use_checkpoint)
        pkl.addWidget(self._use_checkpoint, 3, 0, 1, 3)

        pkl.addWidget(QLabel("Resume File:"), 4, 0)
        self._checkpoint_resume_file = QLineEdit(
            getattr(self._defaults, "checkpoint_resume_file", ""))
        self._checkpoint_resume_file.setPlaceholderText("path/to/checkpoint.pickle")
        self._checkpoint_resume_file.setEnabled(self._defaults.use_checkpoint)
        pkl.addWidget(self._checkpoint_resume_file, 4, 1)
        browse_resume_btn = QPushButton("…")
        browse_resume_btn.setMaximumWidth(30)
        browse_resume_btn.setEnabled(self._defaults.use_checkpoint)
        browse_resume_btn.clicked.connect(self._browse_checkpoint_resume_file)
        pkl.addWidget(browse_resume_btn, 4, 2)
        self._browse_resume_btn = browse_resume_btn   # keep reference to enable/disable

        self._use_checkpoint.toggled.connect(self._checkpoint_resume_file.setEnabled)
        self._use_checkpoint.toggled.connect(self._browse_resume_btn.setEnabled)

        pkg.toggled.connect(self._on_checkpoint_section_toggled)

        layout.addWidget(pkg)

        # Keep references so _on_pipeline_mode_changed can show/hide them.
        # rg (RGI) is intentionally excluded - it stays visible in both modes.
        self._full_pipeline_groups = [tg, cg, tmg, og, sg, qg, csg, pkg]

        # Apply initial visibility/gating for the default (Full) mode.
        self._on_pipeline_mode_changed(self._pipeline_mode.currentIndex())

    # ── Right panel ───────────────────────────────────────────────────────────

    def _build_right_panel(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)

        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Start")
        self._run_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        self._run_btn.clicked.connect(self._start_pipeline)
        btn_row.addWidget(self._run_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_pipeline)
        btn_row.addWidget(self._stop_btn)

        self._results_btn = QPushButton("Results")
        self._results_btn.setToolTip("View results")
        self._results_btn.clicked.connect(self._show_results_page)
        btn_row.addWidget(self._results_btn)

        self._query_nav_btn = QPushButton("Query")
        self._query_nav_btn.setToolTip("Find nearest tree by world coordinate")
        self._query_nav_btn.clicked.connect(self._show_query_page)
        btn_row.addWidget(self._query_nav_btn)

        self._debug_mode = QCheckBox("Debug Mode")
        self._debug_mode.setToolTip(
            "Save intermediate point clouds at each pipeline step.\n"
            "Automatically generates a PDF report on completion.\n"
            "Full pipeline: also enables Save Clusters and Save Leaf Removal Output."
        )
        btn_row.addWidget(self._debug_mode)

        layout.addLayout(btn_row)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("Step %v / %m")
        layout.addWidget(self._progress_bar)

        self._progress_label = QLabel("Ready")
        layout.addWidget(self._progress_label)

        log_layout = QVBoxLayout()
        log_layout.addWidget(QLabel("Pipeline Log:"))
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFontFamily("Courier")
        log_layout.addWidget(self._log, stretch=1)
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self._log.clear)
        log_layout.addWidget(clear_btn)
        layout.addLayout(log_layout, stretch=1)

    # ── Config building ───────────────────────────────────────────────────────

    def _build_config(self) -> EcomodelConfig:
        return EcomodelConfig(
            input_folder=self._input_folder.text().strip(),
            results_folder=self._results_folder.text().strip(),
            cylinder_filename=self._cylinder_filename.text().strip(),
            scalar_field=self._scalar_field.currentText().strip() or "intensity",
            normalize_scalar=self._normalize_scalar.isChecked(),
            cube_size=self._cube_size.value(),
            meter_conversion=self._meter_conversion.value(),
            csf_band_size=self._csf_band_size.value(),
            csf_threshold=self._csf_threshold.value(),
            csf_offset=self._csf_offset.value(),
            remove_under_ground=self._remove_underground.isChecked(),
            terrain_grid_size=self._terrain_grid_size.value(),
            use_denoise=self._use_denoise.isChecked(),
            denoise_grid_size=self._denoise_grid_size.value(),
            denoise_min_points=self._denoise_min_points.value(),
            denoise_resolution=self._denoise_resolution.value(),
            remove_duplicates=self._remove_duplicates.isChecked(),
            segment_intensity_threshold=self._seg_intensity.value(),
            debug_mode=self._debug_mode.isChecked(),
            save_clusters=self._save_clusters.isChecked() or self._debug_mode.isChecked(),
            qsm_intensity_threshold=self._qsm_intensity.value(),
            save_leaf_removal_output=self._save_leaf_output.isChecked() or self._debug_mode.isChecked(),
            run_qsm=self._run_qsm.isChecked(),
            run_leaf_removal=self._run_leaf_removal.isChecked(),
            create_cylinder_plot=self._create_cylinder_plot.isChecked(),
            patch_diam1=self._patch_diam1.value(),
            ball_rad1=self._ball_rad1.value(),
            nmin1=self._nmin1.value(),
            patch_diam2_min=self._patch_diam2_min.value(),
            patch_diam2_max=self._patch_diam2_max.value(),
            ball_rad2=self._ball_rad2.value(),
            rgi_noise_percentile=self._rgi_noise_percentile.value(),
            rgi_angle_deg=self._rgi_angle_deg.value(),
            rgi_curv_thresh=self._rgi_curv_thresh.value(),
            rgi_resid_thresh=self._rgi_resid_thresh.value(),
            rgi_k=self._rgi_k.value(),
            rgi_min_cluster_size=self._rgi_min_cluster_size.value(),
            rgi_max_cluster_size=self._rgi_max_cluster_size.value(),
            rgi_smooth_mode=self._rgi_smooth_mode.isChecked(),
            rgi_use_residual_test=self._rgi_use_residual.isChecked(),
            rgi_use_curvature_test=self._rgi_use_curvature.isChecked(),
            checkpoint_name=self._checkpoint_name.text().strip(),
            checkpoint_folder=self._checkpoint_folder.text().strip(),
            use_checkpoint=self._use_checkpoint.isChecked(),
            save_checkpoint=self._save_checkpoint.isChecked(),
            checkpoint_resume_file=self._checkpoint_resume_file.text().strip(),
            pipeline_type="lite" if self._pipeline_mode.currentIndex() == 1 else "full",
            lite_intensity_threshold=self._lite_intensity.value(),
            lite_csf_cloth_resolution=self._lite_csf_cloth.value(),
            lite_csf_class_threshold=self._lite_csf_thresh.value(),
            lite_csf_iterations=self._lite_csf_iters.value(),
            lite_csf_remove_underground=self._lite_csf_underground.isChecked(),
            lite_noise_voxel_size=self._lite_noise_voxel.value(),
            lite_noise_min_points=self._lite_noise_minpts.value(),
            lite_patch_diam1=self._lite_patch_diam1.value(),
            lite_ball_rad1=self._lite_ball_rad1.value(),
            lite_nmin1=self._lite_nmin1.value(),
            lite_patch_diam2_min=self._lite_patch_diam2_min.value(),
            lite_patch_diam2_max=self._lite_patch_diam2_max.value(),
            lite_ball_rad2=self._lite_ball_rad2.value(),
            lite_segmenter_type="treelearn" if self._lite_segmenter.currentIndex() == 1 else "scanline",
            treelearn_config_path=self._treelearn_config.text().strip(),
            treelearn_use_gpu=self._treelearn_gpu.isChecked(),
            lite_qsm_method="smartqsm" if self._lite_qsm_method.currentIndex() == 1 else "treeqsm",
            smartqsm_dir=self._smartqsm_dir.text().strip(),
            smartqsm_python=self._smartqsm_python.text().strip(),
            smartqsm_config=self._smartqsm_config.text().strip(),
        )

    # ── Pipeline control ──────────────────────────────────────────────────────

    def _start_pipeline(self) -> None:
        config = self._build_config()

        # When resuming from a specific checkpoint file the original LAS/LAZ
        # folder may no longer exist - the pickled Ecomodel already contains
        # all the loaded point data.  Only validate the input folder when a
        # fresh run is needed (not resuming from an explicit checkpoint file).
        _needs_input_folder = not (
            config.use_checkpoint and config.checkpoint_resume_file
        )
        if _needs_input_folder:
            if not config.input_folder:
                QMessageBox.warning(self, "Missing Input", "Please select a LAS/LAZ input folder.")
                return
            if not Path(config.input_folder).is_dir():
                QMessageBox.warning(
                    self,
                    "Folder Not Found",
                    f"Input folder does not exist:\n{config.input_folder}",
                )
                return

        self._log.clear()
        self._progress_bar.setValue(0)
        self._progress_label.setText("Starting...")
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

        self._thread = QThread()
        self._worker = EcomodelWorker(config)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._append_log)
        self._worker.progress.connect(self._update_progress)
        self._worker.tile_update.connect(self._on_tile_update)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_done)

        self._thread.start()

    def _stop_pipeline(self) -> None:
        if self._worker:
            self._worker.request_stop()
            self._stop_btn.setEnabled(False)
            self._append_log("[GUI] Stop requested - pipeline will halt after the current step.\n")

    # ── Reset handlers ────────────────────────────────────────────────────────

    def _reset_subdivision_defaults(self) -> None:
        d = self._defaults
        self._cube_size.setValue(d.cube_size)
        self._meter_conversion.setValue(d.meter_conversion)

    def _reset_csf_defaults(self) -> None:
        d = self._defaults
        self._csf_band_size.setValue(d.csf_band_size)
        self._csf_threshold.setValue(d.csf_threshold)
        self._csf_offset.setValue(d.csf_offset)
        self._remove_underground.setChecked(d.remove_under_ground)

    def _reset_terrain_defaults(self) -> None:
        d = self._defaults
        self._terrain_grid_size.setValue(d.terrain_grid_size)

    def _reset_cleanup_defaults(self) -> None:
        d = self._defaults
        self._remove_duplicates.setChecked(d.remove_duplicates)
        self._use_denoise.setChecked(d.use_denoise)
        self._denoise_grid_size.setValue(d.denoise_grid_size)
        self._denoise_min_points.setValue(d.denoise_min_points)
        self._denoise_resolution.setValue(d.denoise_resolution)

    def _reset_segmentation_defaults(self) -> None:
        d = self._defaults
        self._seg_intensity.setValue(d.segment_intensity_threshold)
        self._save_clusters.setChecked(d.save_clusters)

    def _reset_qsm_defaults(self) -> None:
        d = self._defaults
        self._run_qsm.setChecked(d.run_qsm)
        self._run_leaf_removal.setChecked(getattr(d, "run_leaf_removal", True))
        self._qsm_intensity.setValue(d.qsm_intensity_threshold)
        self._save_leaf_output.setChecked(d.save_leaf_removal_output)
        self._create_cylinder_plot.setChecked(d.create_cylinder_plot)

    def _reset_cover_defaults(self) -> None:
        d = self._defaults
        self._patch_diam1.setValue(d.patch_diam1)
        self._ball_rad1.setValue(d.ball_rad1)
        self._nmin1.setValue(d.nmin1)
        self._patch_diam2_min.setValue(d.patch_diam2_min)
        self._patch_diam2_max.setValue(d.patch_diam2_max)
        self._ball_rad2.setValue(d.ball_rad2)

    def _reset_rgi_defaults(self) -> None:
        d = self._defaults
        self._rgi_noise_percentile.setValue(d.rgi_noise_percentile)
        self._rgi_angle_deg.setValue(d.rgi_angle_deg)
        self._rgi_curv_thresh.setValue(d.rgi_curv_thresh)
        self._rgi_resid_thresh.setValue(d.rgi_resid_thresh)
        self._rgi_k.setValue(d.rgi_k)
        self._rgi_min_cluster_size.setValue(d.rgi_min_cluster_size)
        self._rgi_max_cluster_size.setValue(d.rgi_max_cluster_size)
        self._rgi_smooth_mode.setChecked(d.rgi_smooth_mode)
        self._rgi_use_residual.setChecked(d.rgi_use_residual_test)
        self._rgi_use_curvature.setChecked(d.rgi_use_curvature_test)

    def _on_segmenter_changed(self, index: int) -> None:
        """Show TreeLearn controls only when TreeLearn segmenter is selected."""
        self._treelearn_widget.setVisible(index == 1)

    def _on_lite_qsm_method_changed(self, index: int) -> None:
        """Show SmartQSM path controls; cover sets are TreeQSM-only."""
        self._smartqsm_widget.setVisible(index == 1)
        if self._pipeline_mode.currentIndex() == 1:   # lite mode
            self._lite_cover_group.setVisible(index == 0)

    def _browse_smartqsm_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Select SmartQSM checkout", self._smartqsm_dir.text() or "")
        if d:
            self._smartqsm_dir.setText(d)

    def _browse_smartqsm_python(self) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self, "Select SmartQSM python", self._smartqsm_python.text() or "",
            "Python (python*.exe);;All files (*)")
        if p:
            self._smartqsm_python.setText(p)

    def _browse_smartqsm_config(self) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self, "Select SmartQSM config", self._smartqsm_config.text() or "",
            "YAML (*.yaml *.yml);;All files (*)")
        if p:
            self._smartqsm_config.setText(p)

    def _browse_treelearn_config(self) -> None:
        """Open file picker for TreeLearn YAML config."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select TreeLearn Config YAML",
            self._treelearn_config.text() or "",
            "YAML files (*.yaml *.yml);;All files (*)",
        )
        if path:
            self._treelearn_config.setText(path)

    def _on_pipeline_mode_changed(self, index: int) -> None:
        """Show/hide parameter groups based on selected pipeline mode.

        Lite mode shows: Lite Settings + RGI (shared leaf/wood params).
        Full mode shows: all full-pipeline groups + RGI.
        """
        is_lite = (index == 1)
        for grp in self._full_pipeline_groups:
            grp.setVisible(not is_lite)
        for grp in self._lite_groups:
            grp.setVisible(is_lite)
        if is_lite:
            self._on_lite_qsm_method_changed(self._lite_qsm_method.currentIndex())
        # RGI group is always visible, but only the Lite pipeline consumes these
        # params - gate it in Full mode so the controls don't mislead.
        self._rgi_group.setVisible(True)
        if is_lite:
            self._rgi_group.setEnabled(True)
            self._rgi_group.setTitle("RGI Leaf/Wood Separation")
        else:
            self._rgi_group.setEnabled(False)
            self._rgi_group.setTitle("RGI Leaf/Wood Separation (not active in Full pipeline)")

    def _trigger_report(self) -> None:
        """Generate a PDF report for the most recent run (called automatically in debug mode)."""
        try:
            from ecomodel_report_maker import generate_report
        except ImportError:
            self._append_log(
                "[Report] Skipped - reportlab not installed "
                "(pip install reportlab to enable PDF reports).\n"
            )
            return

        input_folder = self._input_folder.text().strip()
        results_folder = self._results_folder.text().strip()
        if not input_folder or not results_folder:
            return

        if self._last_run_dir and self._last_run_dir.exists():
            report_run_dir = str(self._last_run_dir)
        else:
            report_run_dir = results_folder

        output_pdf = str(Path(results_folder) / "ecomodel_report.pdf")
        self._append_log(f"[Report] Generating PDF report: {output_pdf}\n")

        from gui.worker import BgTask
        task = BgTask(generate_report, input_folder, report_run_dir, output_pdf)
        task.result.connect(self._on_report_done)
        task.error.connect(self._on_report_error)
        task.start()
        self._report_task = task

    def _on_report_done(self, page_count) -> None:
        output_pdf = str(Path(self._results_folder.text().strip()) / "ecomodel_report.pdf")
        self._append_log(f"[Report] Done - {page_count} page(s): {output_pdf}\n")
        self.statusBar().showMessage(f"Report saved: {output_pdf}")
        import os, sys
        if sys.platform == "win32":
            os.startfile(output_pdf)

    def _on_report_error(self, tb: str) -> None:
        self._append_log(f"[Report] ERROR:\n{tb}\n")
        QMessageBox.critical(self, "Report Error", tb[:600])

    def _on_checkpoint_section_toggled(self, checked: bool) -> None:
        """Re-apply the resume-file enable state after the section is unlocked."""
        if checked:
            resume_on = self._use_checkpoint.isChecked()
            self._checkpoint_resume_file.setEnabled(resume_on)
            self._browse_resume_btn.setEnabled(resume_on)

    def _browse_checkpoint_folder(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        folder = QFileDialog.getExistingDirectory(
            self, "Select Checkpoint Folder",
            self._checkpoint_folder.text() or "."
        )
        if folder:
            self._checkpoint_folder.setText(folder)

    def _browse_checkpoint_resume_file(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        start = (
            self._checkpoint_folder.text().strip()
            or self._results_folder.text().strip()
            or "."
        )
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Checkpoint File", start,
            "Pickle files (*.pickle);;All files (*)"
        )
        if not path:
            return
        self._checkpoint_resume_file.setText(path)
        # Auto-populate folder and name from the selected file
        p = Path(path)
        self._checkpoint_folder.setText(str(p.parent))
        stem = p.stem   # e.g. "ecomodel_checkpoint_post_normalize"
        for tag in ("_post_normalize", "_post_segmentation", "_post_qsm"):
            if stem.endswith(tag):
                stem = stem[: -len(tag)]
                break
        self._checkpoint_name.setText(stem)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _append_log(self, msg: str) -> None:
        self._log.insertPlainText(msg)
        self._log.moveCursor(QTextCursor.End)

    def _update_progress(self, current: int, total: int, label: str) -> None:
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(current)
        self._progress_label.setText(label)
        self.statusBar().showMessage(f"Step {current}/{total}: {label}")

    def _on_finished(self, eco) -> None:
        if eco is None:
            self._append_log("[GUI] Pipeline was stopped by user.\n")
            self._progress_label.setText("Stopped")
            self._progress_bar.setValue(0)
            self.statusBar().showMessage("Pipeline stopped")
        else:
            # eco is an Ecomodel object (full pipeline) or a dict (lite pipeline)
            if isinstance(eco, dict):
                run_dir = Path(eco.get("run_dir", self._results_folder.text().strip()))
                results = run_dir.parent.resolve()
                self._last_run_dir = run_dir
            else:
                results = Path(eco.results_folder).resolve()
                self._last_run_dir = None
            self._append_log("[GUI] Pipeline completed successfully.\n")
            self._append_log(f"[GUI] Results written to: {results}\n")
            self._progress_label.setText("Done")
            self.statusBar().showMessage(f"Done - results in {results}")
            if self._debug_mode.isChecked():
                self._trigger_report()
            page = self._show_results_page()
            if self._last_run_dir:
                page.notify_run_complete(self._last_run_dir.parent)
            else:
                page.notify_run_complete(results)

    def _on_error(self, tb: str) -> None:
        self._append_log(f"[ERROR]\n{tb}\n")
        QMessageBox.critical(self, "Pipeline Error", tb[:600])
        self._progress_label.setText("Error")

    def _on_thread_done(self) -> None:
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        if self._thread:
            self._thread.deleteLater()
            self._thread = None
        if self._worker:
            self._worker.deleteLater()
            self._worker = None

    def closeEvent(self, event) -> None:
        """Stop the pipeline worker and drain all background tasks cleanly."""
        # Stop the pipeline if it is running
        if self._worker is not None:
            self._worker.request_stop()
            if self._thread is not None:
                self._thread.quit()
                self._thread.wait(5_000)

        # Cancel and wait for every in-flight BgTask (scan / load / render)
        from gui.worker import cancel_all_bg_tasks
        cancel_all_bg_tasks(3_000)

        super().closeEvent(event)

    def _on_tile_update(self, tile_name: str, status: str, cyl_count: int, output_path: str) -> None:
        parts = [f"[Tile] {tile_name}: {status}"]
        if cyl_count >= 0:
            parts.append(f"{cyl_count} cylinders")
        if output_path:
            parts.append(output_path)
        self._append_log(" - ".join(parts) + "\n")

    # ── Results page navigation ───────────────────────────────────────────────

    def _show_results_page(self) -> ResultsPage:
        """Sync the results folder and flip to the results page."""
        folder = self._results_folder.text().strip()
        self._results_page.set_results_folder(folder)
        self._query_page.set_results_folder(folder)
        self._stack.setCurrentIndex(1)
        return self._results_page

    def _show_query_page(self) -> QueryPage:
        """Sync the results folder and flip to the query page."""
        folder = self._results_folder.text().strip()
        self._query_page.set_results_folder(folder)
        self._stack.setCurrentIndex(2)
        return self._query_page

    # ── File dialogs ──────────────────────────────────────────────────────────

    def _browse_input(self) -> None:
        start = self._input_folder.text().strip() or "."
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a LAS/LAZ File", start,
            "LiDAR files (*.las *.laz);;All files (*)"
        )
        if path:
            folder = str(Path(path).parent)
            self._input_folder.setText(folder)
            self._auto_populate_names(folder)

    def _auto_populate_names(self, folder: str) -> None:
        """
        Derive cylinder filename and checkpoint name from the input folder.

        Uses the LAS/LAZ file stem if there is exactly one file in the folder,
        otherwise uses the folder name itself.  Replaces non-word characters
        with underscores so the names are safe to use as file/folder names.
        """
        import os
        import re
        from pathlib import Path

        p = Path(folder)
        try:
            las_files = [f for f in os.listdir(folder)
                         if f.lower().endswith((".las", ".laz"))]
        except OSError:
            las_files = []

        stem = Path(las_files[0]).stem if len(las_files) == 1 else p.name
        stem = re.sub(r"[^\w]", "_", stem).strip("_") or "ecomodel"

        self._cylinder_filename.setText(stem)
        self._checkpoint_name.setText(f"{stem}_checkpoint")

        self._populate_scalar_fields(folder, las_files)

    def _populate_scalar_fields(self, folder: str, las_files: list) -> None:
        """
        Fill the Scalar Field dropdown from the first LAS/LAZ file's dimensions.

        Preserves the current selection if it is still available; otherwise
        keeps 'intensity'.  Silent on any read error (dropdown stays editable
        so the user can still type a field name).
        """
        if not las_files:
            return
        try:
            from Utils.Utils import list_las_scalar_fields
            import os
            fields = list_las_scalar_fields(os.path.join(folder, las_files[0]))
        except Exception:
            fields = []
        if not fields:
            return

        current = self._scalar_field.currentText().strip() or "intensity"
        self._scalar_field.blockSignals(True)
        self._scalar_field.clear()
        self._scalar_field.addItems(fields)
        # Restore prior selection if still present, else prefer 'intensity'.
        idx = self._scalar_field.findText(current)
        if idx < 0:
            idx = self._scalar_field.findText("intensity")
        self._scalar_field.setCurrentIndex(idx if idx >= 0 else 0)
        self._scalar_field.blockSignals(False)

    def _on_scalar_field_changed(self, text: str) -> None:
        """Auto-enable normalization when a non-intensity field is chosen."""
        if text.strip().lower() != "intensity":
            self._normalize_scalar.setChecked(True)

    def _browse_results(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Results Folder")
        if folder:
            self._results_folder.setText(folder)

    # ── Widget factory helpers ────────────────────────────────────────────────

    @staticmethod
    def _spin(lo: int, hi: int, val: int) -> QSpinBox:
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setValue(val)
        return w

    @staticmethod
    def _dspin(lo: float, hi: float, val: float, dec: int = 3) -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setValue(val)
        w.setDecimals(dec)
        return w