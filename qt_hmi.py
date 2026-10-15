"""Qt version of the laser scanner HMI.

This is intentionally separate from hmi.py while the embedded 3D interface
is being evaluated. It keeps the same camera, Arduino, and scan protocol.
MSD project
"""

from __future__ import annotations

import sys
import time
import os
import ctypes
from pathlib import Path

import cv2
import numpy as np
import serial
import PySide6
# Allow the application to run on systems where the user cannot install the
# small XCB cursor runtime package with sudo. The bundled copy is optional.
local_xcb_cursor = Path(__file__).parent / ".runtime-libs" / "usr" / "lib" / "x86_64-linux-gnu" / "libxcb-cursor.so.0"
if local_xcb_cursor.exists():
    ctypes.CDLL(str(local_xcb_cursor), mode=ctypes.RTLD_GLOBAL)
# OpenCV ships a Qt plugin directory that can shadow PySide6's plugins. Point
# Qt at the matching PySide6 plugins before QApplication is constructed.
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(Path(PySide6.__file__).parent / "Qt" / "plugins")
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
    QPlainTextEdit, QScrollArea, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)
from serial.tools import list_ports
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plyfile import PlyData

from scanner import (
    Settings, kalman_profile, laser_centerline, laser_mask, median_preprocess,
    savitzky_golay_profile, statistical_outlier_removal, triangulate, voxel_downsample,
    write_ply,
)


class QtScanner(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Laser Triangulation Scanner")
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowSystemMenuHint
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        self.resize(1400, 850)
        self.setMinimumSize(1000, 650)
        self.camera: cv2.VideoCapture | None = None
        self.arduino: serial.Serial | None = None
        self.camera_map: dict[str, int] = {}
        self.port_map: dict[str, str] = {}
        self.latest_points = np.empty((0, 3), dtype=np.float32)
        self.latest_frame: np.ndarray | None = None
        self.cloud: list[np.ndarray] = []
        self.raw_cloud: list[np.ndarray] = []
        self.raw_profiles: list[np.ndarray] = []
        self.raw_angles: list[float] = []
        self.scan_angles: list[float] = []
        self.scan_index = 0
        self.scanning = False
        self.model_figure: Figure | None = None
        self.model_axes = None
        self.model_canvas: FigureCanvasQTAgg | None = None

        self.tabs = QTabWidget()
        self.scan_page = self.build_scan_page()
        self.model_page = self.build_model_page()
        self.tabs.addTab(self.scan_page, "Scanner")
        self.tabs.addTab(self.model_page, "3D Model")
        self.setCentralWidget(self.tabs)
        self.refresh_cameras()
        self.refresh_ports()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_preview)
        self.timer.start(30)

    def field(self, value: str) -> QLineEdit:
        field = QLineEdit(value)
        field.setMaximumWidth(120)
        return field

    def build_scan_page(self) -> QWidget:
        page = QWidget()
        root = QHBoxLayout(page)
        controls = QWidget()
        controls.setMinimumWidth(290)
        controls.setMaximumWidth(600)
        controls_layout = QVBoxLayout(controls)
        splitter = QSplitter(Qt.Horizontal)
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        controls_scroll.setFrameShape(QScrollArea.NoFrame)
        controls_scroll.setWidget(controls)
        splitter.addWidget(controls_scroll)

        connections = QGroupBox("Connections")
        grid = QGridLayout(connections)
        self.camera_combo = QComboBox()
        self.port_combo = QComboBox()
        grid.addWidget(QLabel("USB camera"), 0, 0)
        grid.addWidget(self.camera_combo, 0, 1)
        refresh_cam = QPushButton("Refresh")
        refresh_cam.clicked.connect(self.refresh_cameras)
        grid.addWidget(refresh_cam, 0, 2)
        connect_cam = QPushButton("Connect")
        connect_cam.clicked.connect(self.connect_camera)
        grid.addWidget(connect_cam, 1, 1)
        grid.addWidget(QLabel("Arduino"), 2, 0)
        grid.addWidget(self.port_combo, 2, 1)
        refresh_port = QPushButton("Refresh")
        refresh_port.clicked.connect(self.refresh_ports)
        grid.addWidget(refresh_port, 2, 2)
        connect_port = QPushButton("Connect")
        connect_port.clicked.connect(self.connect_arduino)
        grid.addWidget(connect_port, 3, 1)
        controls_layout.addWidget(connections)

        calibration = QGroupBox("Calibration")
        form = QFormLayout(calibration)
        self.height = self.field("300")
        self.angle = self.field("30")
        self.baseline = self.field("100")
        self.focal = self.field("700")
        for label, widget in (("Camera height mm", self.height), ("Laser angle deg", self.angle),
                              ("Laser baseline mm", self.baseline), ("Focal length px", self.focal)):
            form.addRow(label, widget)
        controls_layout.addWidget(calibration)

        motion = QGroupBox("Rack and pinion")
        form = QFormLayout(motion)
        self.start = self.field("0")
        self.end = self.field("180")
        self.step = self.field("5")
        self.travel = self.field("50")
        self.settle = self.field("250")
        self.axis = QComboBox(); self.axis.addItems(["X", "Y", "Z"])
        self.direction = QComboBox(); self.direction.addItems(["+", "-"])
        for label, widget in (("Start servo deg", self.start), ("End servo deg", self.end),
                              ("Step deg", self.step), ("Travel / 180 deg", self.travel),
                              ("Settle ms", self.settle), ("Scan axis", self.axis),
                              ("Direction", self.direction)):
            form.addRow(label, widget)
        controls_layout.addWidget(motion)

        filtering = QGroupBox("Filtering")
        filter_form = QFormLayout(filtering)
        self.filter_combo = QComboBox()
        self.filter_combo.addItems((
            "None (Raw Data)", "Median Filtering", "Subpixel Laser-Line Extraction",
            "Statistical Outlier Removal (SOR)", "Savitzky–Golay Filtering",
            "Voxel Grid Downsampling", "Kalman Filtering",
        ))
        self.filter_combo.currentTextChanged.connect(self.update_filter_controls)
        filter_form.addRow("Technique", self.filter_combo)
        self.filter_params = QWidget()
        self.filter_params_form = QFormLayout(self.filter_params)
        self.filter_params_form.setContentsMargins(0, 0, 0, 0)
        filter_form.addRow(self.filter_params)
        filter_buttons = QHBoxLayout()
        apply_filter = QPushButton("Apply Filter")
        apply_filter.clicked.connect(self.apply_filter)
        reset_filter = QPushButton("Reset to Raw")
        reset_filter.clicked.connect(self.reset_filter)
        filter_buttons.addWidget(apply_filter)
        filter_buttons.addWidget(reset_filter)
        filter_form.addRow(filter_buttons)
        self.filter_stats = QLabel("Raw: 0 | Displayed: 0")
        self.filter_stats.setWordWrap(True)
        filter_form.addRow(self.filter_stats)
        controls_layout.addWidget(filtering)

        actions = QGroupBox("Scan")
        form = QFormLayout(actions)
        self.manual_angle = self.field("0")
        form.addRow("Manual servo deg", self.manual_angle)
        for label, callback in (("Move servo", self.move_manual), ("Capture slice", self.capture_slice),
                                ("Start automatic scan", self.start_scan), ("Stop scan", self.stop_scan),
                                ("Clear cloud", self.clear_cloud), ("Export PLY", self.export_cloud),
                                ("Open 3D model", self.open_model)):
            button = QPushButton(label)
            button.clicked.connect(callback)
            form.addRow(button)
        controls_layout.addWidget(actions)
        self.status = QLabel("Ready")
        self.status.setWordWrap(True)
        controls_layout.addWidget(self.status)
        controls_layout.addStretch()
        self.update_filter_controls(self.filter_combo.currentText())

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.preview = QLabel("Connect a camera to begin")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(650, 450)
        self.preview.setStyleSheet("background: #101316; color: #b0bec5;")
        right_layout.addWidget(self.preview, 1)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        right_layout.addWidget(self.log, 0)
        splitter.addWidget(right)
        splitter.setSizes([350, 900])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)
        return page

    def build_model_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        bar = QHBoxLayout()
        self.model_mode = QComboBox()
        self.model_mode.addItems(["Points", "Surface", "Points + surface"])
        self.model_mode.setCurrentText("Points + surface")
        refresh = QPushButton("Refresh model")
        refresh.clicked.connect(self.open_model)
        open_file = QPushButton("Open PLY file")
        open_file.clicked.connect(self.open_ply)
        bar.addWidget(QLabel("Display:"))
        bar.addWidget(self.model_mode)
        bar.addWidget(refresh)
        bar.addWidget(open_file)
        bar.addStretch()
        layout.addLayout(bar)
        self.model_figure = Figure(figsize=(10, 7), dpi=100)
        self.model_figure.patch.set_facecolor("#20252b")
        self.model_axes = self.model_figure.add_subplot(111, projection="3d")
        self.model_axes.set_facecolor("#101316")
        self.model_canvas = FigureCanvasQTAgg(self.model_figure)
        toolbar = NavigationToolbar2QT(self.model_canvas, page)
        layout.addWidget(toolbar)
        layout.addWidget(self.model_canvas, 1)
        return page

    def log_message(self, message: str) -> None:
        self.log.appendPlainText(f"{time.strftime('%H:%M:%S')}  {message}")

    def refresh_cameras(self) -> None:
        self.camera_map.clear()
        self.camera_combo.clear()
        paths = sorted(Path("/dev").glob("video[0-9]*"), key=lambda p: p.name)
        for path in paths:
            index = int(path.name[5:])
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if cap.isOpened():
                label = f"USB camera - {path}"
                self.camera_map[label] = index
                self.camera_combo.addItem(label)
            cap.release()
        self.log_message(f"Cameras found: {self.camera_combo.count()}")

    def refresh_ports(self) -> None:
        self.port_map.clear()
        self.port_combo.clear()
        for port in list_ports.comports():
            name = (port.description or port.manufacturer or "Serial device").strip()
            label = f"{name} - {port.device}"
            self.port_map[label] = port.device
            self.port_combo.addItem(label)
        self.log_message(f"Serial devices found: {self.port_combo.count()}")

    def connect_camera(self) -> None:
        index = self.camera_map.get(self.camera_combo.currentText())
        if index is None:
            QMessageBox.warning(self, "Camera", "Select a camera first")
            return
        if self.camera is not None:
            self.camera.release()
        self.camera = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.camera.isOpened():
            self.camera = None
            QMessageBox.critical(self, "Camera", "Could not open the selected camera")
            return
        self.log_message(f"Camera connected: {self.camera_combo.currentText()}")
        self.status.setText("Camera connected")

    def connect_arduino(self) -> None:
        device = self.port_map.get(self.port_combo.currentText())
        if not device:
            QMessageBox.warning(self, "Arduino", "Select an Arduino port first")
            return
        try:
            if self.arduino is not None:
                self.arduino.close()
            self.arduino = serial.Serial(device, 9600, timeout=1)
            time.sleep(2)
            self.log_message(f"Arduino connected: {device}")
        except serial.SerialException as error:
            self.arduino = None
            QMessageBox.critical(self, "Arduino", str(error))

    def scanner_settings(self) -> Settings:
        return Settings(float(self.height.text()), float(self.angle.text()),
                        float(self.baseline.text()), float(self.focal.text()))

    def update_preview(self) -> None:
        if self.camera is None or not self.camera.isOpened():
            return
        ok, frame = self.camera.read()
        if not ok:
            return
        self.latest_frame = frame
        try:
            settings = self.scanner_settings()
        except ValueError:
            settings = Settings()
        processing_frame = frame
        method = "median"
        if self.filter_combo.currentText() == "Median Filtering":
            kernel = self.median_kernel.value() if hasattr(self, "median_kernel") else 3
            processing_frame = median_preprocess(frame, kernel)
        elif self.filter_combo.currentText() == "Subpixel Laser-Line Extraction":
            method = "subpixel"
        mask = laser_mask(processing_frame, settings)
        window_radius = self.subpixel_window.value() if hasattr(self, "subpixel_window") else 3
        xs, ys = laser_centerline(mask, settings, processing_frame, method=method,
                                   window_radius=window_radius)
        self.latest_points = triangulate(xs, ys, frame.shape[1], frame.shape[0], settings)
        frame[mask > 0] = (0, 255, 255)
        cv2.putText(frame, f"laser points: {len(self.latest_points)} | cloud: {self.cloud_size()}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        # Keep aspect ratio while filling the preview; cropping removes blank bars.
        self.preview.setPixmap(QPixmap.fromImage(image).scaled(self.preview.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))

    def send_move(self, angle: float) -> bool:
        if self.arduino is None or not self.arduino.is_open:
            self.log_message("Arduino is not connected")
            return False
        command = f"{max(0, min(180, angle)):.0f}\n"
        self.arduino.reset_input_buffer()
        self.arduino.write(command.encode("ascii"))
        reply = self.arduino.readline().decode("ascii", errors="replace").strip()
        self.log_message(f"> {command.strip()}  < {reply or 'no reply'}")
        return reply.startswith("Servo moved to:")

    def move_manual(self) -> None:
        try:
            angle = float(self.manual_angle.text())
            if not 0 <= angle <= 180:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Servo angle", "Enter an angle from 0 to 180")
            return
        self.send_move(angle)

    def displacement(self, angle: float) -> float:
        sign = 1 if self.direction.currentText() == "+" else -1
        return angle / 180 * float(self.travel.text()) * sign

    def capture_slice(self, angle: float | None = None) -> bool:
        if len(self.latest_points) == 0:
            self.status.setText("No laser line detected")
            return False
        try:
            current = float(self.manual_angle.text()) if angle is None else angle
            profile = self.latest_points.copy()
            translated = profile.copy()
            translated[:, "XYZ".index(self.axis.currentText())] += self.displacement(current)
            self.raw_profiles.append(profile)
            self.raw_angles.append(current)
            self.raw_cloud.append(translated)
            self.apply_filter(show_message=False)
            self.status.setText(f"Cloud contains {self.cloud_size()} points")
            return True
        except ValueError:
            QMessageBox.warning(self, "Scan settings", "Check the numeric fields")
            return False

    def start_scan(self) -> None:
        try:
            start, end, step = float(self.start.text()), float(self.end.text()), abs(float(self.step.text()))
            if not 0 <= start <= 180 or not 0 <= end <= 180 or step == 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Scan range", "Use servo angles from 0 to 180 and a non-zero step")
            return
        if self.camera is None or self.arduino is None:
            QMessageBox.warning(self, "Scan", "Connect both the camera and Arduino first")
            return
        self.scan_angles = list(np.arange(start, end + step / 2, step if end >= start else -step))
        self.scan_index = 0
        self.scanning = True
        self.run_scan_step()

    def run_scan_step(self) -> None:
        if not self.scanning or self.scan_index >= len(self.scan_angles):
            self.stop_scan()
            return
        angle = float(self.scan_angles[self.scan_index])
        self.manual_angle.setText(f"{angle:.1f}")
        if not self.send_move(angle):
            self.stop_scan()
            return
        try:
            delay = max(0, int(float(self.settle.text())))
        except ValueError:
            delay = 250
        QTimer.singleShot(delay, lambda: self.capture_and_next(angle))

    def capture_and_next(self, angle: float) -> None:
        if self.scanning:
            self.capture_slice(angle)
            self.scan_index += 1
            self.run_scan_step()

    def stop_scan(self) -> None:
        self.scanning = False
        self.status.setText(f"Scan complete: {self.cloud_size()} points")

    def cloud_size(self) -> int:
        return sum(len(points) for points in self.cloud)

    def clear_cloud(self) -> None:
        self.cloud.clear()
        self.raw_cloud.clear()
        self.raw_profiles.clear()
        self.raw_angles.clear()
        if self.model_axes is not None:
            self.model_axes.clear()
            self.model_axes.set_facecolor("#101316")
            self.model_canvas.draw_idle()
        self.status.setText("Cloud cleared")

    def update_filter_controls(self, technique: str) -> None:
        while self.filter_params_form.count():
            item = self.filter_params_form.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        if technique == "Median Filtering":
            self.median_kernel = QSpinBox()
            self.median_kernel.setRange(3, 7)
            self.median_kernel.setSingleStep(2)
            self.median_kernel.setValue(3)
            self.filter_params_form.addRow("Kernel size", self.median_kernel)
        elif technique == "Subpixel Laser-Line Extraction":
            self.subpixel_window = QSpinBox()
            self.subpixel_window.setRange(1, 12)
            self.subpixel_window.setValue(3)
            self.filter_params_form.addRow("Window radius", self.subpixel_window)
        elif technique == "Statistical Outlier Removal (SOR)":
            self.sor_neighbors = QSpinBox()
            self.sor_neighbors.setRange(1, 200)
            self.sor_neighbors.setValue(20)
            self.sor_ratio = QDoubleSpinBox()
            self.sor_ratio.setRange(0.01, 20.0)
            self.sor_ratio.setSingleStep(0.1)
            self.sor_ratio.setValue(2.0)
            self.filter_params_form.addRow("Neighbors", self.sor_neighbors)
            self.filter_params_form.addRow("Std. ratio", self.sor_ratio)
        elif technique == "Savitzky–Golay Filtering":
            self.savgol_window = QSpinBox()
            self.savgol_window.setRange(3, 999)
            self.savgol_window.setSingleStep(2)
            self.savgol_window.setValue(7)
            self.savgol_order = QSpinBox()
            self.savgol_order.setRange(1, 10)
            self.savgol_order.setValue(2)
            self.filter_params_form.addRow("Window length", self.savgol_window)
            self.filter_params_form.addRow("Polynomial order", self.savgol_order)
        elif technique == "Voxel Grid Downsampling":
            self.voxel_size = QDoubleSpinBox()
            self.voxel_size.setRange(0.001, 1000.0)
            self.voxel_size.setDecimals(3)
            self.voxel_size.setValue(1.0)
            self.filter_params_form.addRow("Voxel size mm", self.voxel_size)
        elif technique == "Kalman Filtering":
            self.kalman_process = QDoubleSpinBox()
            self.kalman_process.setRange(0.0, 10000.0)
            self.kalman_process.setDecimals(5)
            self.kalman_process.setValue(0.01)
            self.kalman_measurement = QDoubleSpinBox()
            self.kalman_measurement.setRange(0.00001, 10000.0)
            self.kalman_measurement.setDecimals(5)
            self.kalman_measurement.setValue(1.0)
            self.filter_params_form.addRow("Process noise", self.kalman_process)
            self.filter_params_form.addRow("Measurement noise", self.kalman_measurement)

    def reset_filter(self) -> None:
        self.filter_combo.setCurrentText("None (Raw Data)")
        self.cloud = [points.copy() for points in self.raw_cloud]
        self.update_filter_stats(0.0)
        self.status.setText("Displaying raw cloud")

    def update_filter_stats(self, elapsed_ms: float) -> None:
        raw_count = sum(len(points) for points in self.raw_cloud)
        displayed_count = self.cloud_size()
        removed = (100.0 * (raw_count - displayed_count) / raw_count) if raw_count else 0.0
        values = np.vstack(self.cloud)[:, 2] if self.cloud and displayed_count else np.empty(0)
        deviation = float(np.std(values)) if len(values) else 0.0
        self.filter_stats.setText(
            f"Technique: {self.filter_combo.currentText()}\n"
            f"Raw: {raw_count} | Displayed: {displayed_count}\n"
            f"Removed: {removed:.2f}% | Time: {elapsed_ms:.1f} ms\n"
            f"Height std. dev.: {deviation:.4f} mm"
        )

    def apply_filter(self, show_message: bool = True) -> None:
        import time as timer
        started = timer.perf_counter()
        technique = self.filter_combo.currentText()
        try:
            if technique == "None (Raw Data)":
                self.cloud = [points.copy() for points in self.raw_cloud]
            elif technique == "Median Filtering" or technique == "Subpixel Laser-Line Extraction":
                # These are acquisition-stage filters; existing captured slices
                # cannot be reconstructed from frames that were not saved.
                self.cloud = [points.copy() for points in self.raw_cloud]
                if show_message and self.raw_cloud:
                    QMessageBox.information(self, "Acquisition filter", "This filter applies to new camera captures.")
            elif technique in ("Savitzky–Golay Filtering", "Kalman Filtering"):
                self.cloud = []
                for profile, angle in zip(self.raw_profiles, self.raw_angles):
                    if technique == "Savitzky–Golay Filtering":
                        filtered = savitzky_golay_profile(profile, self.savgol_window.value(), self.savgol_order.value())
                    else:
                        filtered = kalman_profile(profile, self.kalman_process.value(), self.kalman_measurement.value())
                    translated = filtered.copy()
                    translated[:, "XYZ".index(self.axis.currentText())] += self.displacement(angle)
                    self.cloud.append(translated)
            else:
                raw = np.vstack(self.raw_cloud) if self.raw_cloud else np.empty((0, 3), dtype=np.float32)
                if technique == "Statistical Outlier Removal (SOR)":
                    filtered = statistical_outlier_removal(raw, self.sor_neighbors.value(), self.sor_ratio.value())
                else:
                    filtered = voxel_downsample(raw, self.voxel_size.value())
                self.cloud = [filtered] if len(filtered) else []
            self.update_filter_stats((timer.perf_counter() - started) * 1000.0)
            self.status.setText(f"{technique} applied to {self.cloud_size()} points")
        except (ValueError, RuntimeError, ImportError) as error:
            self.cloud = [points.copy() for points in self.raw_cloud]
            self.update_filter_stats((timer.perf_counter() - started) * 1000.0)
            self.log_message(f"Filter failed: {error}")
            if show_message:
                QMessageBox.warning(self, "Filter", str(error))

    def open_model(self) -> None:
        if not self.cloud:
            QMessageBox.warning(self, "3D model", "Capture a scan first")
            return
        points = np.vstack(self.cloud).astype(np.float32)
        self.render_model(points)
        self.tabs.setCurrentWidget(self.model_page)
        self.status.setText("3D model ready")

    def open_ply(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open PLY point cloud", "", "PLY files (*.ply)")
        if not path:
            return
        try:
            ply = PlyData.read(path)
            vertices = ply["vertex"].data
            points = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float32)
            faces = []
            if "face" in ply:
                for row in ply["face"].data:
                    indices = row["vertex_indices"]
                    if len(indices) >= 3:
                        faces.append(points[np.asarray(indices, dtype=np.int64)])
            self.render_model(points, faces)
            self.tabs.setCurrentWidget(self.model_page)
            self.status.setText(f"Loaded {len(points)} vertices, {len(faces)} faces")
            self.log_message(f"Loaded PLY: {path} ({len(points)} vertices, {len(faces)} faces)")
        except Exception as error:
            QMessageBox.critical(self, "Open PLY", str(error))
            self.log_message(f"PLY load failed: {error}")

    def render_model(self, points: np.ndarray, faces: list[np.ndarray] | None = None) -> None:
        axis = self.model_axes
        axis.clear()
        axis.set_facecolor("#101316")
        mode = self.model_mode.currentText()
        faces = faces or []
        if mode in ("Points", "Points + surface"):
            axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=3, c="#39d6c4", depthshade=True)
        if mode in ("Surface", "Points + surface"):
            if faces:
                axis.add_collection3d(Poly3DCollection(faces, facecolors="#4d9dcc", alpha=0.58,
                                                        edgecolors="#b8e3ef", linewidths=0.08))
            elif len(points) >= 30:
                try:
                    scan_axis = "XYZ".index(self.axis.currentText())
                    other_axes = [index for index in range(3) if index != scan_axis]
                    other_axis = max(other_axes, key=lambda index: np.ptp(points[:, index]))
                    remaining_axis = 3 - scan_axis - other_axis
                    axis.plot_trisurf(points[:, scan_axis], points[:, other_axis], points[:, remaining_axis],
                                      color="#4d9dcc", alpha=0.58, linewidth=0.08, edgecolor="#b8e3ef")
                except Exception as error:
                    self.log_message(f"Surface reconstruction skipped: {error}")
        axis.set_xlabel("X (mm)")
        axis.set_ylabel("Y (mm)")
        axis.set_zlabel("Z (mm)")
        axis.view_init(elev=25, azim=-60)
        ranges = np.ptp(points, axis=0)
        ranges[ranges == 0] = 1
        axis.set_box_aspect(ranges)
        self.model_canvas.draw_idle()

    def export_cloud(self) -> None:
        if not self.cloud:
            QMessageBox.warning(self, "Export", "Capture a scan first")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export point cloud", "scan.ply", "PLY files (*.ply)")
        if path:
            write_ply(Path(path), np.vstack(self.cloud))
            self.status.setText(f"Saved {self.cloud_size()} points")

    def closeEvent(self, event) -> None:
        self.timer.stop()
        if self.camera is not None:
            self.camera.release()
        if self.arduino is not None:
            self.arduino.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = QtScanner()
    window.show()
    sys.exit(app.exec())
