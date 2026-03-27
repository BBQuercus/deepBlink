"""Napari dock widget for interactive spot labeling.

Provides a smooth workflow for:
1. Opening a folder of images
2. Navigating through images one-by-one
3. Running preliminary spot detection (deepBlink model or LoG)
4. Editing spots interactively (add/move/delete via napari Points layer)
5. Saving labeled spots to CSV format compatible with ``deepblink create``
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QKeySequence
    from qtpy.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QProgressBar,
        QPushButton,
        QShortcut,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
except ImportError as e:
    raise ImportError(
        "napari and qtpy are required for the labeling plugin. "
        'Install with: pip install "deepblink[napari]"'
    ) from e

from ..io import EXTENSIONS, load_image

# Default point display settings for comfortable labeling
_POINT_SIZE = 10
_POINT_EDGE_WIDTH = 1.5
_POINT_FACE_COLOR = "#FF4081"
_POINT_EDGE_COLOR = "#FFFFFF"


class SpotLabeler(QWidget):
    """Main dock widget for the spot labeling workflow.

    Parameters
    ----------
    napari_viewer : napari.Viewer
        The napari viewer instance this widget is docked into.
    """

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # State
        self._image_paths: List[Path] = []
        self._current_index: int = -1
        self._labels: Dict[str, np.ndarray] = {}  # basename -> (N, 2) r,c coords
        self._output_dir: Optional[Path] = None
        self._model = None
        self._points_layer = None
        self._image_layer = None

        self._build_ui()
        self._connect_signals()
        self._bind_shortcuts()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(6)

        # --- Folder selection ---
        grp_folder = QGroupBox("Image Folder")
        fl = QVBoxLayout()
        row = QHBoxLayout()
        self._btn_open = QPushButton("Open Folder...")
        self._lbl_folder = QLabel("No folder selected")
        self._lbl_folder.setWordWrap(True)
        row.addWidget(self._btn_open)
        row.addWidget(self._lbl_folder, stretch=1)
        fl.addLayout(row)

        # Output directory
        row_out = QHBoxLayout()
        self._btn_output = QPushButton("Output Folder...")
        self._lbl_output = QLabel("Same as image folder")
        self._lbl_output.setWordWrap(True)
        row_out.addWidget(self._btn_output)
        row_out.addWidget(self._lbl_output, stretch=1)
        fl.addLayout(row_out)
        grp_folder.setLayout(fl)
        layout.addWidget(grp_folder)

        # --- Image list & navigation ---
        grp_nav = QGroupBox("Images")
        nl = QVBoxLayout()
        self._list_images = QListWidget()
        self._list_images.setMaximumHeight(160)
        nl.addWidget(self._list_images)

        nav_row = QHBoxLayout()
        self._btn_prev = QPushButton("< Prev (A)")
        self._btn_next = QPushButton("Next (D) >")
        self._lbl_counter = QLabel("0 / 0")
        self._lbl_counter.setAlignment(Qt.AlignCenter)
        nav_row.addWidget(self._btn_prev)
        nav_row.addWidget(self._lbl_counter, stretch=1)
        nav_row.addWidget(self._btn_next)
        nl.addLayout(nav_row)

        self._progress = QProgressBar()
        self._progress.setFormat("Labeled: %v / %m")
        nl.addWidget(self._progress)

        self._chk_autosave = QCheckBox("Auto-save on navigate")
        self._chk_autosave.setChecked(True)
        nl.addWidget(self._chk_autosave)

        grp_nav.setLayout(nl)
        layout.addWidget(grp_nav)

        # --- Detection settings ---
        grp_detect = QGroupBox("Spot Detection")
        dl = QVBoxLayout()

        # Method selector
        row_method = QHBoxLayout()
        row_method.addWidget(QLabel("Method:"))
        self._combo_method = QComboBox()
        self._combo_method.addItems(["LoG (scikit-image)", "deepBlink model"])
        row_method.addWidget(self._combo_method, stretch=1)
        dl.addLayout(row_method)

        # LoG parameters
        self._grp_log = QWidget()
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(0, 0, 0, 0)

        row_sigma = QHBoxLayout()
        row_sigma.addWidget(QLabel("Min sigma:"))
        self._spin_min_sigma = QDoubleSpinBox()
        self._spin_min_sigma.setRange(0.1, 50.0)
        self._spin_min_sigma.setValue(1.0)
        self._spin_min_sigma.setSingleStep(0.5)
        row_sigma.addWidget(self._spin_min_sigma)
        row_sigma.addWidget(QLabel("Max sigma:"))
        self._spin_max_sigma = QDoubleSpinBox()
        self._spin_max_sigma.setRange(0.1, 50.0)
        self._spin_max_sigma.setValue(5.0)
        self._spin_max_sigma.setSingleStep(0.5)
        row_sigma.addWidget(self._spin_max_sigma)
        log_layout.addLayout(row_sigma)

        row_log2 = QHBoxLayout()
        row_log2.addWidget(QLabel("Num sigma:"))
        self._spin_num_sigma = QSpinBox()
        self._spin_num_sigma.setRange(1, 50)
        self._spin_num_sigma.setValue(10)
        row_log2.addWidget(self._spin_num_sigma)
        row_log2.addWidget(QLabel("Threshold:"))
        self._spin_threshold = QDoubleSpinBox()
        self._spin_threshold.setRange(0.001, 1.0)
        self._spin_threshold.setValue(0.05)
        self._spin_threshold.setSingleStep(0.01)
        self._spin_threshold.setDecimals(3)
        row_log2.addWidget(self._spin_threshold)
        log_layout.addLayout(row_log2)
        self._grp_log.setLayout(log_layout)
        dl.addWidget(self._grp_log)

        # Model parameters
        self._grp_model = QWidget()
        model_layout = QVBoxLayout()
        model_layout.setContentsMargins(0, 0, 0, 0)
        row_model = QHBoxLayout()
        self._btn_load_model = QPushButton("Load Model (.h5)...")
        self._lbl_model = QLabel("No model loaded")
        row_model.addWidget(self._btn_load_model)
        row_model.addWidget(self._lbl_model, stretch=1)
        model_layout.addLayout(row_model)

        row_prob = QHBoxLayout()
        row_prob.addWidget(QLabel("Probability:"))
        self._spin_prob = QDoubleSpinBox()
        self._spin_prob.setRange(0.01, 0.99)
        self._spin_prob.setValue(0.50)
        self._spin_prob.setSingleStep(0.05)
        row_prob.addWidget(self._spin_prob)
        model_layout.addLayout(row_prob)
        self._grp_model.setLayout(model_layout)
        self._grp_model.setVisible(False)
        dl.addWidget(self._grp_model)

        self._btn_detect = QPushButton("Detect Spots (R)")
        self._btn_detect.setStyleSheet("font-weight: bold;")
        dl.addWidget(self._btn_detect)
        grp_detect.setLayout(dl)
        layout.addWidget(grp_detect)

        # --- Editing hints ---
        grp_edit = QGroupBox("Editing (napari Points layer)")
        el = QVBoxLayout()
        hint = QLabel(
            "After detection, the Spots layer is active:\n"
            "  - Press 2: Add mode (click to place spots)\n"
            "  - Press 3: Select mode (drag to move)\n"
            "  - Select + Delete/Backspace: remove\n"
            "\n"
            "Keyboard shortcuts:\n"
            "  A / D  - Previous / Next image\n"
            "  R      - Run detection\n"
            "  Ctrl+S - Save current image"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #AAAAAA; font-size: 11px;")
        el.addWidget(hint)
        grp_edit.setLayout(el)
        layout.addWidget(grp_edit)

        # --- Save ---
        grp_save = QGroupBox("Save")
        sl = QVBoxLayout()
        row_save = QHBoxLayout()
        self._btn_save = QPushButton("Save Current (Ctrl+S)")
        self._btn_save_all = QPushButton("Save All Labeled")
        row_save.addWidget(self._btn_save)
        row_save.addWidget(self._btn_save_all)
        sl.addLayout(row_save)

        self._lbl_status = QLabel("")
        self._lbl_status.setWordWrap(True)
        self._lbl_status.setStyleSheet("color: #4CAF50;")
        sl.addWidget(self._lbl_status)
        grp_save.setLayout(sl)
        layout.addWidget(grp_save)

        layout.addStretch()
        self.setLayout(layout)
        self.setMinimumWidth(320)

    def _connect_signals(self):
        self._btn_open.clicked.connect(self._on_open_folder)
        self._btn_output.clicked.connect(self._on_select_output)
        self._btn_prev.clicked.connect(self._on_prev)
        self._btn_next.clicked.connect(self._on_next)
        self._btn_detect.clicked.connect(self._on_detect)
        self._btn_load_model.clicked.connect(self._on_load_model)
        self._btn_save.clicked.connect(self._on_save_current)
        self._btn_save_all.clicked.connect(self._on_save_all)
        self._combo_method.currentIndexChanged.connect(self._on_method_changed)
        self._list_images.currentRowChanged.connect(self._on_list_selection)

    def _bind_shortcuts(self):
        """Bind keyboard shortcuts for fast navigation and detection."""
        # Navigate: A = previous, D = next (WASD-style, avoids arrow key conflicts)
        self._shortcut_prev = QShortcut(QKeySequence("A"), self.viewer.window._qt_window)
        self._shortcut_prev.activated.connect(self._on_prev)

        self._shortcut_next = QShortcut(QKeySequence("D"), self.viewer.window._qt_window)
        self._shortcut_next.activated.connect(self._on_next)

        # Detect: R = run detection
        self._shortcut_detect = QShortcut(
            QKeySequence("R"), self.viewer.window._qt_window
        )
        self._shortcut_detect.activated.connect(self._on_detect)

        # Save: Ctrl+S
        self._shortcut_save = QShortcut(
            QKeySequence("Ctrl+S"), self.viewer.window._qt_window
        )
        self._shortcut_save.activated.connect(self._on_save_current)

    # ------------------------------------------------------------------
    # Folder / file handling
    # ------------------------------------------------------------------

    def _on_open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return
        self._load_folder(Path(folder))

    def _load_folder(self, folder: Path):
        """Load all images from a folder."""
        self._lbl_folder.setText(str(folder))

        # Set output dir to same folder by default
        if self._output_dir is None:
            self._output_dir = folder
            self._lbl_output.setText(str(folder))

        # Collect image files (deduplicate across case variants)
        paths = []
        for ext in EXTENSIONS:
            paths.extend(folder.glob(f"*.{ext}"))
            paths.extend(folder.glob(f"*.{ext.upper()}"))
        self._image_paths = sorted(set(paths))

        if not self._image_paths:
            self._lbl_status.setText("No images found in folder.")
            self._lbl_status.setStyleSheet("color: #FF5252;")
            return

        # Check for existing label CSVs
        self._scan_existing_labels()

        # Populate list widget
        self._refresh_image_list()
        self._progress.setMaximum(len(self._image_paths))
        self._update_progress()

        # Load first image
        self._navigate_to(0)

    def _on_select_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self._output_dir = Path(folder)
            self._lbl_output.setText(str(self._output_dir))

    def _scan_existing_labels(self):
        """Load existing CSV labels from the output directory."""
        if self._output_dir is None:
            return
        for path in self._image_paths:
            csv_path = self._csv_path_for(path)
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path, index_col=0)
                    if "X" in df.columns and "Y" in df.columns:
                        coords = df[["Y", "X"]].values.astype(np.float64)  # r, c
                        self._labels[path.stem] = coords
                except Exception:
                    pass

    def _csv_path_for(self, image_path: Path) -> Path:
        """Return the CSV path for a given image."""
        out = self._output_dir if self._output_dir else image_path.parent
        return out / f"{image_path.stem}.csv"

    # ------------------------------------------------------------------
    # Image list & navigation
    # ------------------------------------------------------------------

    def _refresh_image_list(self):
        self._list_images.blockSignals(True)
        self._list_images.clear()
        for i, p in enumerate(self._image_paths):
            is_labeled = p.stem in self._labels
            is_current = i == self._current_index
            # Show status prefix for quick scanning
            prefix = ""
            if is_current:
                prefix = ">> "
            elif is_labeled:
                prefix = "   "
            else:
                prefix = "   "
            item = QListWidgetItem(f"{prefix}{p.name}")
            if is_current:
                item.setForeground(Qt.cyan)
            elif is_labeled:
                item.setForeground(Qt.green)
            self._list_images.addItem(item)
        if 0 <= self._current_index < len(self._image_paths):
            self._list_images.setCurrentRow(self._current_index)
        self._list_images.blockSignals(False)

    def _update_progress(self):
        labeled = sum(1 for p in self._image_paths if p.stem in self._labels)
        self._progress.setValue(labeled)
        total = len(self._image_paths)
        self._lbl_counter.setText(
            f"{self._current_index + 1} / {total}"
            if total > 0
            else "0 / 0"
        )

    def _on_list_selection(self, row):
        if row >= 0 and row != self._current_index:
            self._navigate_to(row)

    def _on_prev(self):
        if self._current_index > 0:
            self._navigate_to(self._current_index - 1)

    def _on_next(self):
        if self._current_index < len(self._image_paths) - 1:
            self._navigate_to(self._current_index + 1)

    def _navigate_to(self, index: int):
        """Save current spots (if any), then load the image at *index*."""
        # Auto-save current spots before navigating away
        if self._current_index >= 0 and self._points_layer is not None:
            self._store_current_spots()
            if self._chk_autosave.isChecked():
                self._save_csv(self._image_paths[self._current_index])

        self._current_index = index
        path = self._image_paths[index]

        # Load image
        image = load_image(str(path), is_rgb=False)

        # Update or create image layer
        if self._image_layer is not None and self._image_layer in self.viewer.layers:
            self._image_layer.data = image
            self._image_layer.name = path.name
        else:
            self._image_layer = self.viewer.add_image(
                image, name=path.name, colormap="gray"
            )

        # Load or clear points
        coords = self._labels.get(path.stem, np.empty((0, 2)))
        self._set_points(coords)

        self.viewer.reset_view()
        self._refresh_image_list()
        self._update_progress()
        self._lbl_status.setText("")

    def _store_current_spots(self):
        """Snapshot the current Points layer data into our labels dict."""
        if self._points_layer is None:
            return
        if self._current_index < 0:
            return
        data = np.asarray(self._points_layer.data)
        path = self._image_paths[self._current_index]
        if len(data) > 0:
            self._labels[path.stem] = data.copy()
        elif path.stem in self._labels:
            del self._labels[path.stem]

    def _set_points(self, coords: np.ndarray):
        """Update or create the Points layer with given coordinates."""
        if coords.ndim == 1 and len(coords) == 0:
            coords = np.empty((0, 2))

        if self._points_layer is not None and self._points_layer in self.viewer.layers:
            self._points_layer.data = coords
        else:
            self._points_layer = self.viewer.add_points(
                coords,
                name="Spots",
                size=_POINT_SIZE,
                face_color=_POINT_FACE_COLOR,
                edge_color=_POINT_EDGE_COLOR,
                edge_width=_POINT_EDGE_WIDTH,
                ndim=2,
            )
        # Activate the points layer for immediate editing
        self.viewer.layers.selection.active = self._points_layer

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _on_method_changed(self, idx):
        self._grp_log.setVisible(idx == 0)
        self._grp_model.setVisible(idx == 1)

    def _on_load_model(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Select deepBlink model", "", "HDF5 Files (*.h5)"
        )
        if not fname:
            return
        try:
            from ..io import load_model

            self._model = load_model(fname)
            self._lbl_model.setText(Path(fname).name)
            self._lbl_status.setText("Model loaded.")
            self._lbl_status.setStyleSheet("color: #4CAF50;")
        except Exception as exc:
            self._lbl_status.setText(f"Model load failed: {exc}")
            self._lbl_status.setStyleSheet("color: #FF5252;")

    def _on_detect(self):
        if self._current_index < 0:
            return
        if self._image_layer is None:
            return

        image = self._image_layer.data
        method_idx = self._combo_method.currentIndex()

        self._lbl_status.setText("Detecting...")
        self._lbl_status.setStyleSheet("color: #FFC107;")
        # Force UI repaint before blocking computation
        from qtpy.QtWidgets import QApplication

        QApplication.processEvents()

        if method_idx == 0:
            coords = self._detect_log(image)
        else:
            coords = self._detect_deepblink(image)

        if coords is not None:
            self._set_points(coords)
            self._lbl_status.setText(f"Detected {len(coords)} spots.")
            self._lbl_status.setStyleSheet("color: #4CAF50;")

    def _detect_log(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect spots using Laplacian of Gaussian."""
        from skimage.feature import blob_log

        min_sigma = self._spin_min_sigma.value()
        max_sigma = self._spin_max_sigma.value()
        num_sigma = self._spin_num_sigma.value()
        threshold = self._spin_threshold.value()

        # Normalize for better detection
        img = image.astype(np.float64)
        if img.std() > 0:
            img = (img - img.mean()) / img.std()

        blobs = blob_log(
            img,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            threshold=threshold,
        )
        # blob_log returns (r, c, sigma) - take only r, c
        if len(blobs) > 0:
            return blobs[:, :2]
        return np.empty((0, 2))

    def _detect_deepblink(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect spots using a loaded deepBlink model."""
        if self._model is None:
            self._lbl_status.setText("No model loaded. Load a .h5 model first.")
            self._lbl_status.setStyleSheet("color: #FF5252;")
            return None

        try:
            from ..inference import predict

            probability = self._spin_prob.value()
            coords = predict(image, self._model, probability=probability)
            # predict returns (N, 3) with probability column when probability is set;
            # we only need the first two columns (r, c)
            if len(coords) > 0 and coords.shape[1] > 2:
                coords = coords[:, :2]
            return coords
        except Exception as exc:
            self._lbl_status.setText(f"Prediction failed: {exc}")
            self._lbl_status.setStyleSheet("color: #FF5252;")
            return None

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def _save_csv(self, image_path: Path):
        """Save spots for one image to CSV in FIJI format.

        Output format (compatible with ``deepblink create``)::

            ,X,Y
            0,228.635,63.999
            1,134.339,104.726
        """
        basename = image_path.stem
        if basename not in self._labels:
            return
        coords = self._labels[basename]
        if len(coords) == 0:
            return

        # FIJI format: index, X (column), Y (row)
        df = pd.DataFrame({"X": coords[:, 1], "Y": coords[:, 0]})
        csv_path = self._csv_path_for(image_path)
        df.to_csv(csv_path)

    def _on_save_current(self):
        if self._current_index < 0:
            return
        self._store_current_spots()
        path = self._image_paths[self._current_index]
        self._save_csv(path)
        self._refresh_image_list()
        self._update_progress()
        n = len(self._labels.get(path.stem, []))
        self._lbl_status.setText(f"Saved {n} spots to {path.stem}.csv")
        self._lbl_status.setStyleSheet("color: #4CAF50;")

    def _on_save_all(self):
        self._store_current_spots()
        count = 0
        for path in self._image_paths:
            if path.stem in self._labels:
                self._save_csv(path)
                count += 1
        self._refresh_image_list()
        self._update_progress()
        self._lbl_status.setText(f"Saved labels for {count} images.")
        self._lbl_status.setStyleSheet("color: #4CAF50;")


def launch(path: Optional[str] = None):
    """Launch napari with the SpotLabeler widget.

    Parameters
    ----------
    path : str, optional
        Path to a folder of images to open immediately.

    Examples
    --------
    From Python::

        from deepblink.napari_plugin import launch
        launch("/path/to/images")

    From command line::

        deepblink label -i /path/to/images
    """
    import napari

    viewer = napari.Viewer()
    widget = SpotLabeler(viewer)
    viewer.window.add_dock_widget(widget, name="deepBlink Labeler", area="right")

    if path is not None:
        folder = Path(path)
        if folder.is_dir():
            widget._output_dir = folder
            widget._load_folder(folder)

    napari.run()
