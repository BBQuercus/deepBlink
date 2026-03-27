"""Napari dock widget for interactive prediction visualization.

Post-training tool for checking model performance on arbitrary images:
- Load images of any size and dimensionality
- Auto-detect or manually specify axis arrangement (c, t, z, y, x)
- Navigate channels, z-slices, and timepoints with sliders
- Run prediction on individual slices or all at once
- Adjust probability threshold interactively to tune results
- Spots colored by prediction confidence
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

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
        QLineEdit,
        QPushButton,
        QShortcut,
        QSlider,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
except ImportError as e:
    raise ImportError(
        "napari and qtpy are required for the viewer plugin. "
        'Install with: pip install "deepblink[napari]"'
    ) from e

from ..io import EXTENSIONS, load_image
from ..util import predict_shape

# Point display for prediction overlay
_POINT_SIZE = 10
_POINT_EDGE_WIDTH = 1.5
_POINT_EDGE_COLOR = "#FFFFFF"


class PredictionViewer(QWidget):
    """Dock widget for visualizing deepBlink predictions on arbitrary images.

    Parameters
    ----------
    napari_viewer : napari.Viewer
        The napari viewer instance this widget is docked into.
    """

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # State
        self._model = None
        self._raw_image: Optional[np.ndarray] = None  # full nd image as loaded
        self._image_path: Optional[Path] = None
        self._shape_order: List[str] = []  # e.g. ["c", "z", "y", "x"]
        self._arranged: Optional[np.ndarray] = None  # rearranged to (c, t, z, y, x)
        self._predictions: Dict[Tuple[int, int, int], np.ndarray] = {}  # (c,t,z) -> coords
        self._image_layer = None
        self._points_layer = None

        self._build_ui()
        self._connect_signals()
        self._bind_shortcuts()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(6)

        # --- Model ---
        grp_model = QGroupBox("Model")
        ml = QVBoxLayout()
        row_model = QHBoxLayout()
        self._btn_load_model = QPushButton("Load Model (.h5)...")
        self._lbl_model = QLabel("No model loaded")
        row_model.addWidget(self._btn_load_model)
        row_model.addWidget(self._lbl_model, stretch=1)
        ml.addLayout(row_model)

        row_prob = QHBoxLayout()
        row_prob.addWidget(QLabel("Probability:"))
        self._spin_prob = QDoubleSpinBox()
        self._spin_prob.setRange(0.01, 0.99)
        self._spin_prob.setValue(0.50)
        self._spin_prob.setSingleStep(0.05)
        row_prob.addWidget(self._spin_prob)
        ml.addLayout(row_prob)
        grp_model.setLayout(ml)
        layout.addWidget(grp_model)

        # --- Image ---
        grp_image = QGroupBox("Image")
        il = QVBoxLayout()
        row_img = QHBoxLayout()
        self._btn_load_image = QPushButton("Open Image...")
        self._btn_load_folder = QPushButton("Open Folder...")
        row_img.addWidget(self._btn_load_image)
        row_img.addWidget(self._btn_load_folder)
        il.addLayout(row_img)

        self._lbl_image = QLabel("No image loaded")
        self._lbl_image.setWordWrap(True)
        il.addWidget(self._lbl_image)

        # Shape / axis arrangement
        row_shape = QHBoxLayout()
        row_shape.addWidget(QLabel("Shape:"))
        self._edit_shape = QLineEdit()
        self._edit_shape.setPlaceholderText("auto-detected, e.g. c,z,y,x")
        row_shape.addWidget(self._edit_shape, stretch=1)
        self._btn_apply_shape = QPushButton("Apply")
        row_shape.addWidget(self._btn_apply_shape)
        il.addLayout(row_shape)

        self._lbl_shape_info = QLabel("")
        self._lbl_shape_info.setWordWrap(True)
        self._lbl_shape_info.setStyleSheet("color: #AAAAAA; font-size: 11px;")
        il.addWidget(self._lbl_shape_info)

        self._chk_rgb = QCheckBox("Image is RGB (convert to grayscale)")
        il.addWidget(self._chk_rgb)

        grp_image.setLayout(il)
        layout.addWidget(grp_image)

        # --- Folder navigation ---
        self._grp_folder_nav = QGroupBox("Folder Navigation")
        fnl = QVBoxLayout()
        nav_row = QHBoxLayout()
        self._btn_img_prev = QPushButton("< Prev (A)")
        self._btn_img_next = QPushButton("Next (D) >")
        self._lbl_img_counter = QLabel("0 / 0")
        self._lbl_img_counter.setAlignment(Qt.AlignCenter)
        nav_row.addWidget(self._btn_img_prev)
        nav_row.addWidget(self._lbl_img_counter, stretch=1)
        nav_row.addWidget(self._btn_img_next)
        fnl.addLayout(nav_row)
        self._grp_folder_nav.setLayout(fnl)
        self._grp_folder_nav.setVisible(False)
        layout.addWidget(self._grp_folder_nav)

        # --- Dimension sliders ---
        grp_dims = QGroupBox("Dimensions")
        dl = QVBoxLayout()

        # Channel slider
        row_c = QHBoxLayout()
        row_c.addWidget(QLabel("Channel:"))
        self._slider_c = QSlider(Qt.Horizontal)
        self._slider_c.setMinimum(0)
        self._slider_c.setMaximum(0)
        row_c.addWidget(self._slider_c, stretch=1)
        self._lbl_c = QLabel("0")
        self._lbl_c.setMinimumWidth(24)
        row_c.addWidget(self._lbl_c)
        dl.addLayout(row_c)

        # Time slider
        row_t = QHBoxLayout()
        row_t.addWidget(QLabel("Time:"))
        self._slider_t = QSlider(Qt.Horizontal)
        self._slider_t.setMinimum(0)
        self._slider_t.setMaximum(0)
        row_t.addWidget(self._slider_t, stretch=1)
        self._lbl_t = QLabel("0")
        self._lbl_t.setMinimumWidth(24)
        row_t.addWidget(self._lbl_t)
        dl.addLayout(row_t)

        # Z slider
        row_z = QHBoxLayout()
        row_z.addWidget(QLabel("Z-slice:"))
        self._slider_z = QSlider(Qt.Horizontal)
        self._slider_z.setMinimum(0)
        self._slider_z.setMaximum(0)
        row_z.addWidget(self._slider_z, stretch=1)
        self._lbl_z = QLabel("0")
        self._lbl_z.setMinimumWidth(24)
        row_z.addWidget(self._lbl_z)
        dl.addLayout(row_z)

        grp_dims.setLayout(dl)
        layout.addWidget(grp_dims)

        # --- Predict ---
        grp_predict = QGroupBox("Predict")
        pl = QVBoxLayout()
        row_btns = QHBoxLayout()
        self._btn_predict_slice = QPushButton("Predict Current (R)")
        self._btn_predict_slice.setStyleSheet("font-weight: bold;")
        self._btn_predict_all = QPushButton("Predict All Slices")
        row_btns.addWidget(self._btn_predict_slice)
        row_btns.addWidget(self._btn_predict_all)
        pl.addLayout(row_btns)

        # Live threshold
        row_thresh = QHBoxLayout()
        row_thresh.addWidget(QLabel("Filter threshold:"))
        self._slider_thresh = QSlider(Qt.Horizontal)
        self._slider_thresh.setMinimum(1)
        self._slider_thresh.setMaximum(99)
        self._slider_thresh.setValue(50)
        row_thresh.addWidget(self._slider_thresh, stretch=1)
        self._lbl_thresh = QLabel("0.50")
        self._lbl_thresh.setMinimumWidth(32)
        row_thresh.addWidget(self._lbl_thresh)
        pl.addLayout(row_thresh)

        self._chk_color_prob = QCheckBox("Color spots by probability")
        self._chk_color_prob.setChecked(True)
        pl.addWidget(self._chk_color_prob)

        grp_predict.setLayout(pl)
        layout.addWidget(grp_predict)

        # --- Status ---
        self._lbl_status = QLabel("")
        self._lbl_status.setWordWrap(True)
        self._lbl_status.setStyleSheet("color: #4CAF50;")
        layout.addWidget(self._lbl_status)

        # --- Stats ---
        self._lbl_stats = QLabel("")
        self._lbl_stats.setWordWrap(True)
        self._lbl_stats.setStyleSheet("color: #AAAAAA; font-size: 11px;")
        layout.addWidget(self._lbl_stats)

        layout.addStretch()
        self.setLayout(layout)
        self.setMinimumWidth(320)

    def _connect_signals(self):
        self._btn_load_model.clicked.connect(self._on_load_model)
        self._btn_load_image.clicked.connect(self._on_load_image)
        self._btn_load_folder.clicked.connect(self._on_load_folder)
        self._btn_apply_shape.clicked.connect(self._on_apply_shape)
        self._btn_img_prev.clicked.connect(self._on_img_prev)
        self._btn_img_next.clicked.connect(self._on_img_next)
        self._btn_predict_slice.clicked.connect(self._on_predict_slice)
        self._btn_predict_all.clicked.connect(self._on_predict_all)
        self._slider_c.valueChanged.connect(self._on_dim_changed)
        self._slider_t.valueChanged.connect(self._on_dim_changed)
        self._slider_z.valueChanged.connect(self._on_dim_changed)
        self._slider_thresh.valueChanged.connect(self._on_thresh_changed)
        self._chk_color_prob.stateChanged.connect(self._on_dim_changed)
        self._spin_prob.valueChanged.connect(self._on_prob_changed)

    def _bind_shortcuts(self):
        self._shortcut_prev = QShortcut(QKeySequence("A"), self.viewer.window._qt_window)
        self._shortcut_prev.activated.connect(self._on_img_prev)
        self._shortcut_next = QShortcut(QKeySequence("D"), self.viewer.window._qt_window)
        self._shortcut_next.activated.connect(self._on_img_next)
        self._shortcut_predict = QShortcut(
            QKeySequence("R"), self.viewer.window._qt_window
        )
        self._shortcut_predict.activated.connect(self._on_predict_slice)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

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
            self._set_status("Model loaded.", ok=True)
        except Exception as exc:
            self._set_status(f"Model load failed: {exc}", ok=False)

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _on_load_image(self):
        exts = " ".join(f"*.{e}" for e in EXTENSIONS)
        fname, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", f"Images ({exts})"
        )
        if not fname:
            return
        self._folder_paths = []
        self._folder_index = -1
        self._grp_folder_nav.setVisible(False)
        self._open_image(Path(fname))

    def _on_load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return
        folder = Path(folder)
        paths = []
        for ext in EXTENSIONS:
            paths.extend(folder.glob(f"*.{ext}"))
            paths.extend(folder.glob(f"*.{ext.upper()}"))
        self._folder_paths = sorted(set(paths))
        if not self._folder_paths:
            self._set_status("No images found in folder.", ok=False)
            return
        self._folder_index = 0
        self._grp_folder_nav.setVisible(True)
        self._open_image(self._folder_paths[0])
        self._update_folder_counter()

    def _on_img_prev(self):
        if not hasattr(self, "_folder_paths") or not self._folder_paths:
            return
        if self._folder_index > 0:
            self._folder_index -= 1
            self._open_image(self._folder_paths[self._folder_index])
            self._update_folder_counter()

    def _on_img_next(self):
        if not hasattr(self, "_folder_paths") or not self._folder_paths:
            return
        if self._folder_index < len(self._folder_paths) - 1:
            self._folder_index += 1
            self._open_image(self._folder_paths[self._folder_index])
            self._update_folder_counter()

    def _update_folder_counter(self):
        total = len(self._folder_paths)
        self._lbl_img_counter.setText(f"{self._folder_index + 1} / {total}")

    def _open_image(self, path: Path):
        """Load an image, detect shape, set up dimension sliders."""
        is_rgb = self._chk_rgb.isChecked()
        self._raw_image = load_image(str(path), is_rgb=is_rgb)
        self._image_path = path
        self._predictions.clear()

        shape = self._raw_image.shape
        self._lbl_image.setText(f"{path.name}  |  shape: {shape}")

        # Auto-detect shape order
        detected = predict_shape(shape)
        self._edit_shape.setText(detected)
        self._lbl_shape_info.setText(
            f"Detected: {detected} for shape {shape}. Edit and click Apply to override."
        )
        self._apply_shape(detected)

    def _on_apply_shape(self):
        text = self._edit_shape.text().strip()
        if text:
            self._apply_shape(text)

    def _apply_shape(self, shape_str: str):
        """Parse shape string, rearrange image to (c, t, z, y, x), update sliders."""
        for c in ["(", ")", " "]:
            shape_str = shape_str.replace(c, "")
        shape_list = shape_str.split(",")
        self._shape_order = shape_list

        image = self._raw_image
        order = ["c", "t", "z", "y", "x"]

        # Expand missing dims
        for dim in order:
            if dim not in shape_list:
                image = np.expand_dims(image, axis=-1)
                shape_list.append(dim)

        # Rearrange to standard order
        for destination, name in enumerate(order):
            source = shape_list.index(name)
            image = np.moveaxis(image, source, destination)
            shape_list.insert(destination, shape_list.pop(source))

        self._arranged = image  # (c, t, z, y, x)

        # Update sliders
        nc, nt, nz = image.shape[0], image.shape[1], image.shape[2]
        self._slider_c.setMaximum(max(0, nc - 1))
        self._slider_t.setMaximum(max(0, nt - 1))
        self._slider_z.setMaximum(max(0, nz - 1))
        self._slider_c.setValue(0)
        self._slider_t.setValue(0)
        self._slider_z.setValue(0)
        self._slider_c.setEnabled(nc > 1)
        self._slider_t.setEnabled(nt > 1)
        self._slider_z.setEnabled(nz > 1)

        self._lbl_shape_info.setText(
            f"Arranged: c={nc}, t={nt}, z={nz}, "
            f"y={image.shape[3]}, x={image.shape[4]}"
        )

        # Display first slice
        self._show_current_slice()

    # ------------------------------------------------------------------
    # Dimension navigation
    # ------------------------------------------------------------------

    def _current_indices(self) -> Tuple[int, int, int]:
        return (self._slider_c.value(), self._slider_t.value(), self._slider_z.value())

    def _on_dim_changed(self):
        c, t, z = self._current_indices()
        self._lbl_c.setText(str(c))
        self._lbl_t.setText(str(t))
        self._lbl_z.setText(str(z))
        self._show_current_slice()

    def _show_current_slice(self):
        """Display the 2D slice for the current c/t/z indices and overlay predictions."""
        if self._arranged is None:
            return
        c, t, z = self._current_indices()
        slice_2d = self._arranged[c, t, z]  # (y, x)

        # Update or create image layer
        if self._image_layer is not None and self._image_layer in self.viewer.layers:
            self._image_layer.data = slice_2d
            name = self._image_path.name if self._image_path else "image"
            self._image_layer.name = f"{name} [c={c} t={t} z={z}]"
        else:
            name = self._image_path.name if self._image_path else "image"
            self._image_layer = self.viewer.add_image(
                slice_2d, name=f"{name} [c={c} t={t} z={z}]", colormap="gray"
            )

        # Overlay predictions if available
        self._update_points_overlay()
        self.viewer.reset_view()

    def _update_points_overlay(self):
        """Show/update the points layer for the current slice's predictions."""
        key = self._current_indices()
        coords_with_prob = self._predictions.get(key)

        if coords_with_prob is None or len(coords_with_prob) == 0:
            coords = np.empty((0, 2))
            colors = np.empty((0, 4))
            n_shown = 0
        else:
            # Filter by threshold
            threshold = self._slider_thresh.value() / 100.0
            if coords_with_prob.shape[1] >= 3:
                mask = coords_with_prob[:, 2] >= threshold
                filtered = coords_with_prob[mask]
                coords = filtered[:, :2]
                probs = filtered[:, 2]
            else:
                coords = coords_with_prob[:, :2]
                probs = np.ones(len(coords))

            n_shown = len(coords)

            # Color by probability
            if self._chk_color_prob.isChecked() and len(probs) > 0:
                colors = _prob_to_rgba(probs)
            else:
                colors = np.tile([1.0, 0.25, 0.5, 1.0], (len(coords), 1))

        if self._points_layer is not None and self._points_layer in self.viewer.layers:
            self._points_layer.data = coords
            if len(colors) > 0:
                self._points_layer.face_color = colors
        else:
            self._points_layer = self.viewer.add_points(
                coords,
                name="Predictions",
                size=_POINT_SIZE,
                face_color=colors if len(colors) > 0 else "#FF4081",
                edge_color=_POINT_EDGE_COLOR,
                edge_width=_POINT_EDGE_WIDTH,
                ndim=2,
            )

        # Update stats
        total = len(coords_with_prob) if coords_with_prob is not None else 0
        c, t, z = key
        if total > 0:
            all_probs = coords_with_prob[:, 2] if coords_with_prob.shape[1] >= 3 else None
            stats = f"Slice c={c} t={t} z={z}: {n_shown}/{total} spots shown"
            if all_probs is not None:
                stats += (
                    f"\nProbability: min={all_probs.min():.3f}, "
                    f"mean={all_probs.mean():.3f}, max={all_probs.max():.3f}"
                )
            self._lbl_stats.setText(stats)
        else:
            n_predicted = sum(
                1 for v in self._predictions.values() if len(v) > 0
            )
            n_total = (
                (self._slider_c.maximum() + 1)
                * (self._slider_t.maximum() + 1)
                * (self._slider_z.maximum() + 1)
            )
            self._lbl_stats.setText(
                f"No prediction for this slice. "
                f"Predicted: {n_predicted}/{n_total} slices"
            )

    # ------------------------------------------------------------------
    # Threshold
    # ------------------------------------------------------------------

    def _on_thresh_changed(self, value):
        self._lbl_thresh.setText(f"{value / 100:.2f}")
        self._update_points_overlay()

    def _on_prob_changed(self):
        """Sync filter threshold slider when the probability spinbox changes."""
        val = int(self._spin_prob.value() * 100)
        self._slider_thresh.setValue(val)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _on_predict_slice(self):
        if self._model is None:
            self._set_status("Load a model first.", ok=False)
            return
        if self._arranged is None:
            self._set_status("Load an image first.", ok=False)
            return

        self._set_status("Predicting...", ok=None)
        from qtpy.QtWidgets import QApplication

        QApplication.processEvents()

        c, t, z = self._current_indices()
        self._predict_single(c, t, z)
        self._update_points_overlay()

        key = (c, t, z)
        n = len(self._predictions.get(key, []))
        self._set_status(f"Detected {n} spots at c={c} t={t} z={z}.", ok=True)

    def _on_predict_all(self):
        if self._model is None:
            self._set_status("Load a model first.", ok=False)
            return
        if self._arranged is None:
            self._set_status("Load an image first.", ok=False)
            return

        nc = self._slider_c.maximum() + 1
        nt = self._slider_t.maximum() + 1
        nz = self._slider_z.maximum() + 1
        total = nc * nt * nz

        self._set_status(f"Predicting all {total} slices...", ok=None)
        from qtpy.QtWidgets import QApplication

        QApplication.processEvents()

        count = 0
        for ci in range(nc):
            for ti in range(nt):
                for zi in range(nz):
                    self._predict_single(ci, ti, zi)
                    count += 1
                    if count % 5 == 0:
                        self._set_status(
                            f"Predicting... {count}/{total}", ok=None
                        )
                        QApplication.processEvents()

        self._update_points_overlay()

        total_spots = sum(len(v) for v in self._predictions.values())
        self._set_status(
            f"Done. {total_spots} spots across {total} slices.", ok=True
        )

    def _predict_single(self, c: int, t: int, z: int):
        """Run prediction on a single 2D slice and store result with probabilities."""
        from ..inference import predict

        slice_2d = self._arranged[c, t, z]
        probability = self._spin_prob.value()
        # Always request probabilities so we can filter interactively
        coords = predict(slice_2d, self._model, probability=probability)
        # coords is (N, 3) with prob column when probability is set, else (N, 2)
        if len(coords) == 0:
            coords = np.empty((0, 3))
        elif coords.shape[1] == 2:
            # Add probability column of 1.0 if not present
            probs = np.ones((len(coords), 1))
            coords = np.hstack([coords, probs])
        self._predictions[(c, t, z)] = coords

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, text: str, ok: Optional[bool]):
        self._lbl_status.setText(text)
        if ok is True:
            self._lbl_status.setStyleSheet("color: #4CAF50;")
        elif ok is False:
            self._lbl_status.setStyleSheet("color: #FF5252;")
        else:
            self._lbl_status.setStyleSheet("color: #FFC107;")


def _prob_to_rgba(probs: np.ndarray) -> np.ndarray:
    """Map probabilities [0, 1] to an RGBA colormap (blue=low, red=high)."""
    probs = np.clip(probs, 0, 1)
    r = probs
    g = 0.2 * (1 - probs)
    b = 1 - probs
    a = np.ones_like(probs)
    return np.column_stack([r, g, b, a])


def launch_viewer(path: Optional[str] = None, model: Optional[str] = None):
    """Launch napari with the PredictionViewer widget.

    Parameters
    ----------
    path : str, optional
        Path to an image or folder of images.
    model : str, optional
        Path to a deepBlink .h5 model file.

    Examples
    --------
    From Python::

        from deepblink.napari_plugin.viewer import launch_viewer
        launch_viewer("/path/to/image.tif", model="model.h5")

    From command line::

        deepblink view -i /path/to/images -m model.h5
    """
    import napari

    viewer = napari.Viewer()
    widget = PredictionViewer(viewer)
    viewer.window.add_dock_widget(widget, name="deepBlink Viewer", area="right")

    # Pre-load model if given
    if model is not None:
        model_path = Path(model)
        if model_path.is_file():
            try:
                from ..io import load_model

                widget._model = load_model(str(model_path))
                widget._lbl_model.setText(model_path.name)
                widget._set_status("Model loaded.", ok=True)
            except Exception as exc:
                widget._set_status(f"Model load failed: {exc}", ok=False)

    # Pre-load image(s) if given
    if path is not None:
        p = Path(path)
        if p.is_file():
            widget._folder_paths = []
            widget._folder_index = -1
            widget._open_image(p)
        elif p.is_dir():
            paths = []
            for ext in EXTENSIONS:
                paths.extend(p.glob(f"*.{ext}"))
                paths.extend(p.glob(f"*.{ext.upper()}"))
            widget._folder_paths = sorted(set(paths))
            if widget._folder_paths:
                widget._folder_index = 0
                widget._grp_folder_nav.setVisible(True)
                widget._open_image(widget._folder_paths[0])
                widget._update_folder_counter()

    napari.run()
