#!/usr/bin/env python3
"""Qt UI for automatic and manual Super 8 frame normalization."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QRect, QSize, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from detect_white_rectangle import (
    CropBox,
    crop_and_normalize,
    image_paths_from_folder,
    load_image,
    output_path_for_image,
    parse_search_area,
    process_image,
)


APP_TITLE = "Super 8 Normalizer"
SETTINGS_PATH = Path(__file__).with_name("qt_normalizer_settings.json")


def bgr_to_qimage(image: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    bytes_per_line = channels * width
    return QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()


def clamp_crop(crop: CropBox, image_width: int, image_height: int) -> CropBox:
    width = max(1, min(crop.width, image_width))
    height = max(1, min(crop.height, image_height))
    x = min(max(0, crop.x), image_width - width)
    y = min(max(0, crop.y), image_height - height)
    return CropBox(x=x, y=y, width=width, height=height)


class ImageCanvas(QWidget):
    cropChanged = pyqtSignal(object)
    searchAreaChanged = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(520, 360)
        self.setMouseTracking(True)
        self._image: QImage | None = None
        self._image_size = QSize(0, 0)
        self._crop: CropBox | None = None
        self._search_area: CropBox | None = None
        self._perforation: dict[str, Any] | None = None
        self._drag_mode: str | None = None
        self._drag_start_image = QPoint()
        self._drag_start_crop: CropBox | None = None
        self._drag_start_search_area: CropBox | None = None
        self._aspect_ratio = 4 / 3
        self._lock_aspect = True
        self._edit_search_area = False
        self._handle_size = 9

    def set_image(self, image: np.ndarray | None) -> None:
        if image is None:
            self._image = None
            self._image_size = QSize(0, 0)
        else:
            self._image = bgr_to_qimage(image)
            self._image_size = QSize(image.shape[1], image.shape[0])
        self.update()

    def set_crop(self, crop: CropBox | None) -> None:
        self._crop = crop
        self.update()

    def set_search_area(self, search_area: CropBox | None) -> None:
        self._search_area = search_area
        self.update()

    def set_perforation(self, perforation: dict[str, Any] | None) -> None:
        self._perforation = perforation
        self.update()

    def set_aspect_ratio(self, aspect_ratio: float) -> None:
        self._aspect_ratio = aspect_ratio

    def set_lock_aspect(self, lock_aspect: bool) -> None:
        self._lock_aspect = lock_aspect

    def set_edit_search_area(self, edit_search_area: bool) -> None:
        self._edit_search_area = edit_search_area
        if not edit_search_area:
            self.unsetCursor()
        self.update()

    def scaled_image_rect(self) -> QRect:
        if self._image is None:
            return QRect()
        scaled = self._image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        return QRect(x, y, scaled.width(), scaled.height())

    def image_to_widget_rect(self, crop: CropBox) -> QRect:
        image_rect = self.scaled_image_rect()
        if image_rect.isEmpty() or self._image_size.isEmpty():
            return QRect()
        sx = image_rect.width() / self._image_size.width()
        sy = image_rect.height() / self._image_size.height()
        x = image_rect.x() + round(crop.x * sx)
        y = image_rect.y() + round(crop.y * sy)
        width = round(crop.width * sx)
        height = round(crop.height * sy)
        return QRect(x, y, width, height)

    def search_handle_rects(self) -> dict[str, QRect]:
        if self._search_area is None:
            return {}
        rect = self.image_to_widget_rect(self._search_area)
        if rect.isEmpty():
            return {}
        size = self._handle_size
        half = size // 2
        points = {
            "nw": rect.topLeft(),
            "n": QPoint(rect.center().x(), rect.top()),
            "ne": rect.topRight(),
            "e": QPoint(rect.right(), rect.center().y()),
            "se": rect.bottomRight(),
            "s": QPoint(rect.center().x(), rect.bottom()),
            "sw": rect.bottomLeft(),
            "w": QPoint(rect.left(), rect.center().y()),
        }
        return {name: QRect(point.x() - half, point.y() - half, size, size) for name, point in points.items()}

    def search_handle_at(self, point: QPoint) -> str | None:
        for name, rect in self.search_handle_rects().items():
            if rect.contains(point):
                return name
        if self._search_area is None:
            return None
        rect = self.image_to_widget_rect(self._search_area).adjusted(-4, -4, 4, 4)
        if not rect.contains(point):
            return None
        edge_margin = 8
        handles = ""
        if abs(point.y() - rect.top()) <= edge_margin:
            handles += "n"
        elif abs(point.y() - rect.bottom()) <= edge_margin:
            handles += "s"
        if abs(point.x() - rect.left()) <= edge_margin:
            handles += "w"
        elif abs(point.x() - rect.right()) <= edge_margin:
            handles += "e"
        return handles or None

    def update_search_cursor(self, point: QPoint) -> None:
        if not self._edit_search_area or self._image is None:
            self.unsetCursor()
            return
        handle = self.search_handle_at(point)
        if handle in {"nw", "se"}:
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif handle in {"ne", "sw"}:
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif handle in {"n", "s"}:
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        elif handle in {"e", "w"}:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif self._search_area is not None and self.image_to_widget_rect(self._search_area).contains(point):
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)

    def widget_to_image_point(self, point: QPoint) -> QPoint:
        image_rect = self.scaled_image_rect()
        if image_rect.isEmpty() or self._image_size.isEmpty():
            return QPoint(0, 0)
        x = round((point.x() - image_rect.x()) * self._image_size.width() / image_rect.width())
        y = round((point.y() - image_rect.y()) * self._image_size.height() / image_rect.height())
        x = min(max(0, x), self._image_size.width() - 1)
        y = min(max(0, y), self._image_size.height() - 1)
        return QPoint(x, y)

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if self._image is None:
            painter.setPen(Qt.GlobalColor.lightGray)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open a folder or image")
            return

        image_rect = self.scaled_image_rect()
        painter.drawPixmap(image_rect, QPixmap.fromImage(self._image))

        if self._crop is not None:
            crop_rect = self.image_to_widget_rect(self._crop)
            painter.setPen(QPen(Qt.GlobalColor.green, 2))
            painter.drawRect(crop_rect)

        if self._search_area is not None:
            search_rect = self.image_to_widget_rect(self._search_area)
            painter.setPen(QPen(Qt.GlobalColor.yellow, 2))
            painter.drawRect(search_rect)
            for handle_rect in self.search_handle_rects().values():
                painter.fillRect(handle_rect, Qt.GlobalColor.yellow)

    def mousePressEvent(self, event: Any) -> None:
        if event.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton) or self._image is None:
            return
        image_point = self.widget_to_image_point(event.position().toPoint())
        self._drag_start_image = image_point
        self._drag_start_crop = self._crop
        self._drag_start_search_area = self._search_area

        editing_search = self._edit_search_area or event.button() == Qt.MouseButton.RightButton
        if editing_search:
            search_rect = self.image_to_widget_rect(self._search_area) if self._search_area else QRect()
            handle = self.search_handle_at(event.position().toPoint())
            if handle is not None:
                self._drag_mode = f"resize_search:{handle}"
            elif search_rect.contains(event.position().toPoint()):
                self._drag_mode = "move_search"
            else:
                self._drag_mode = "draw_search"
                self._search_area = CropBox(image_point.x(), image_point.y(), 1, 1)
                self.searchAreaChanged.emit(self._search_area)
                self.update()
            return

        crop_rect = self.image_to_widget_rect(self._crop) if self._crop else QRect()
        if crop_rect.contains(event.position().toPoint()):
            self._drag_mode = "move"
        else:
            self._drag_mode = "draw"
            self._crop = CropBox(image_point.x(), image_point.y(), 1, 1)
            self.cropChanged.emit(self._crop)
            self.update()

    def mouseMoveEvent(self, event: Any) -> None:
        if self._drag_mode is None:
            self.update_search_cursor(event.position().toPoint())
            return
        if self._image is None:
            return
        point = self.widget_to_image_point(event.position().toPoint())
        image_width = self._image_size.width()
        image_height = self._image_size.height()

        if self._drag_mode.startswith("resize_search:") and self._drag_start_search_area is not None:
            handle = self._drag_mode.split(":", maxsplit=1)[1]
            x1 = self._drag_start_search_area.x
            y1 = self._drag_start_search_area.y
            x2 = self._drag_start_search_area.x2
            y2 = self._drag_start_search_area.y2
            if "w" in handle:
                x1 = min(point.x(), x2)
            if "e" in handle:
                x2 = max(point.x(), x1)
            if "n" in handle:
                y1 = min(point.y(), y2)
            if "s" in handle:
                y2 = max(point.y(), y1)
            self._search_area = clamp_crop(
                CropBox(x1, y1, max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)),
                image_width,
                image_height,
            )
            self.searchAreaChanged.emit(self._search_area)
            self.update()
            return

        if self._drag_mode == "move_search" and self._drag_start_search_area is not None:
            dx = point.x() - self._drag_start_image.x()
            dy = point.y() - self._drag_start_image.y()
            self._search_area = clamp_crop(
                CropBox(
                    self._drag_start_search_area.x + dx,
                    self._drag_start_search_area.y + dy,
                    self._drag_start_search_area.width,
                    self._drag_start_search_area.height,
                ),
                image_width,
                image_height,
            )
            self.searchAreaChanged.emit(self._search_area)
            self.update()
            return

        if self._drag_mode == "draw_search":
            x1 = min(self._drag_start_image.x(), point.x())
            y1 = min(self._drag_start_image.y(), point.y())
            x2 = max(self._drag_start_image.x(), point.x())
            y2 = max(self._drag_start_image.y(), point.y())
            self._search_area = clamp_crop(CropBox(x1, y1, max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)), image_width, image_height)
            self.searchAreaChanged.emit(self._search_area)
            self.update()
            return

        if self._drag_mode == "move" and self._drag_start_crop is not None:
            dx = point.x() - self._drag_start_image.x()
            dy = point.y() - self._drag_start_image.y()
            self._crop = clamp_crop(
                CropBox(
                    self._drag_start_crop.x + dx,
                    self._drag_start_crop.y + dy,
                    self._drag_start_crop.width,
                    self._drag_start_crop.height,
                ),
                image_width,
                image_height,
            )
        else:
            x1 = min(self._drag_start_image.x(), point.x())
            y1 = min(self._drag_start_image.y(), point.y())
            x2 = max(self._drag_start_image.x(), point.x())
            y2 = max(self._drag_start_image.y(), point.y())
            width = max(1, x2 - x1 + 1)
            height = max(1, y2 - y1 + 1)
            if self._lock_aspect:
                height = max(1, round(width / self._aspect_ratio))
            self._crop = clamp_crop(CropBox(x1, y1, width, height), image_width, image_height)

        self.cropChanged.emit(self._crop)
        self.update()

    def mouseReleaseEvent(self, event: Any) -> None:
        self._drag_mode = None


class ImagePreview(QLabel):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self._image: QImage | None = None
        self._message = text
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background: #111; color: #ddd;")

    def set_cv_image(self, image: np.ndarray) -> None:
        self._image = bgr_to_qimage(image)
        self._message = ""
        self.update_pixmap()

    def set_message(self, message: str) -> None:
        self._image = None
        self._message = message
        self.clear()
        self.setText(message)

    def update_pixmap(self) -> None:
        if self._image is None:
            self.setText(self._message)
            return
        self.setPixmap(
            QPixmap.fromImage(self._image).scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self.update_pixmap()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1320, 820)
        self.input_paths: list[Path] = []
        self.output_dir = Path("out-ui")
        self.results: dict[Path, dict[str, Any]] = {}
        self.current_image: np.ndarray | None = None
        self.current_path: Path | None = None
        self.current_crop: CropBox | None = None
        self.current_search_area: CropBox | None = None
        self.saved_search_area: CropBox | None = None
        self.batch_index = 0
        self.batch_running = False
        self.batch_paused = False
        self.batch_timer = QTimer(self)
        self.batch_timer.setSingleShot(True)
        self.batch_timer.timeout.connect(self.process_next_batch_frame)

        self.canvas = ImageCanvas()
        self.debug_preview = ImagePreview("Debug preview")
        self.normalized_preview = ImagePreview("Normalized preview")

        self.frame_list = QListWidget()
        self.frame_list.currentRowChanged.connect(self.select_frame)

        self.output_width = QSpinBox()
        self.output_width.setRange(16, 12000)
        self.output_width.setValue(1440)
        self.output_height = QSpinBox()
        self.output_height.setRange(16, 12000)
        self.output_height.setValue(1080)
        self.jpeg_quality = QSpinBox()
        self.jpeg_quality.setRange(1, 100)
        self.jpeg_quality.setValue(100)
        self.gap_pixels = QSpinBox()
        self.gap_pixels.setRange(0, 500)
        self.gap_pixels.setValue(28)
        self.crop_scale = QSpinBox()
        self.crop_scale.setRange(80, 140)
        self.crop_scale.setValue(108)
        self.y_offset = QSpinBox()
        self.y_offset.setRange(-50, 50)
        self.y_offset.setValue(0)
        self.search_x1 = QSpinBox()
        self.search_x2 = QSpinBox()
        self.search_y1 = QSpinBox()
        self.search_y2 = QSpinBox()
        for spin in (self.search_x1, self.search_x2, self.search_y1, self.search_y2):
            spin.setRange(0, 20000)
            spin.valueChanged.connect(self.search_spins_changed)
        self.color_mode = QComboBox()
        self.color_mode.addItems(["auto", "color", "grayscale"])
        self.mirror_horizontal = QCheckBox()
        self.lock_aspect = QCheckBox()
        self.lock_aspect.setChecked(True)
        self.lock_aspect.toggled.connect(self.canvas.set_lock_aspect)
        self.edit_search_area = QCheckBox()
        self.edit_search_area.toggled.connect(self.canvas.set_edit_search_area)

        self.crop_x = QSpinBox()
        self.crop_y = QSpinBox()
        self.crop_w = QSpinBox()
        self.crop_h = QSpinBox()
        for spin in (self.crop_x, self.crop_y, self.crop_w, self.crop_h):
            spin.setRange(-20000, 20000)
            spin.valueChanged.connect(self.crop_spins_changed)

        self.canvas.cropChanged.connect(self.canvas_crop_changed)
        self.canvas.searchAreaChanged.connect(self.canvas_search_area_changed)

        self.build_ui()
        self.setStatusBar(QStatusBar())
        self.load_settings()
        self.connect_settings_change_handlers()
        self.update_batch_buttons()

    def build_ui(self) -> None:
        self.open_folder_button = QPushButton("Open Folder")
        self.open_folder_button.clicked.connect(self.open_folder)
        self.open_image_button = QPushButton("Open Image")
        self.open_image_button.clicked.connect(self.open_image)
        self.choose_output_button = QPushButton("Output Folder")
        self.choose_output_button.clicked.connect(self.choose_output_folder)
        self.save_settings_button = QPushButton("Save Settings")
        self.save_settings_button.clicked.connect(self.save_settings)
        self.detect_current_button = QPushButton("Auto Current")
        self.detect_current_button.clicked.connect(self.auto_current)
        self.save_manual_button = QPushButton("Save Manual")
        self.save_manual_button.clicked.connect(self.save_manual)
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_batch)
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_batch)
        self.resume_button = QPushButton("Resume")
        self.resume_button.clicked.connect(self.resume_batch)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_batch)

        top_buttons = QHBoxLayout()
        for button in (
            self.open_folder_button,
            self.open_image_button,
            self.choose_output_button,
            self.save_settings_button,
            self.detect_current_button,
            self.save_manual_button,
            self.start_button,
            self.pause_button,
            self.resume_button,
            self.stop_button,
        ):
            top_buttons.addWidget(button)

        options = QFormLayout()
        options.addRow("Width", self.output_width)
        options.addRow("Height", self.output_height)
        options.addRow("JPEG quality", self.jpeg_quality)
        options.addRow("Gap pixels", self.gap_pixels)
        options.addRow("Crop scale %", self.crop_scale)
        options.addRow("Y offset %", self.y_offset)
        options.addRow("Search x1", self.search_x1)
        options.addRow("Search x2", self.search_x2)
        options.addRow("Search y1", self.search_y1)
        options.addRow("Search y2", self.search_y2)
        options.addRow("Edit search", self.edit_search_area)
        options.addRow("Color", self.color_mode)
        options.addRow("Mirror", self.mirror_horizontal)
        options.addRow("Lock 4:3", self.lock_aspect)
        options.addRow("Crop x", self.crop_x)
        options.addRow("Crop y", self.crop_y)
        options.addRow("Crop width", self.crop_w)
        options.addRow("Crop height", self.crop_h)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addLayout(top_buttons)
        left_layout.addWidget(self.frame_list, 1)
        left_layout.addLayout(options)

        previews = QSplitter(Qt.Orientation.Horizontal)
        previews.addWidget(self.canvas)
        previews.addWidget(self.debug_preview)
        previews.addWidget(self.normalized_preview)
        previews.setStretchFactor(0, 3)
        previews.setStretchFactor(1, 2)
        previews.setStretchFactor(2, 2)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(previews)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        self.addAction(quit_action)

    def connect_settings_change_handlers(self) -> None:
        for spin in (
            self.output_width,
            self.output_height,
            self.jpeg_quality,
            self.gap_pixels,
            self.crop_scale,
            self.y_offset,
            self.search_x1,
            self.search_x2,
            self.search_y1,
            self.search_y2,
        ):
            spin.valueChanged.connect(self.remember_search_area_from_widgets)
        self.color_mode.currentTextChanged.connect(self.save_settings)
        self.mirror_horizontal.toggled.connect(self.save_settings)
        self.lock_aspect.toggled.connect(self.save_settings)
        self.edit_search_area.toggled.connect(self.save_settings)

    def settings_data(self) -> dict[str, Any]:
        search_area = self.current_search_area or self.saved_search_area
        return {
            "output_dir": str(self.output_dir),
            "output_width": self.output_width.value(),
            "output_height": self.output_height.value(),
            "jpeg_quality": self.jpeg_quality.value(),
            "gap_pixels": self.gap_pixels.value(),
            "crop_scale_percent": self.crop_scale.value(),
            "y_offset_percent": self.y_offset.value(),
            "color_mode": self.color_mode.currentText(),
            "mirror_horizontal": self.mirror_horizontal.isChecked(),
            "lock_aspect": self.lock_aspect.isChecked(),
            "edit_search": self.edit_search_area.isChecked(),
            "search_area": search_area.to_log_dict() if search_area is not None else None,
        }

    def save_settings(self) -> None:
        SETTINGS_PATH.write_text(json.dumps(self.settings_data(), indent=2) + "\n", encoding="utf-8")
        if self.statusBar() is not None:
            self.statusBar().showMessage(f"Settings saved: {SETTINGS_PATH}")

    def load_settings(self) -> None:
        if not SETTINGS_PATH.exists():
            return
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            logging.warning("Could not load Qt settings from %s: %s", SETTINGS_PATH, error)
            return

        self.output_dir = Path(str(data.get("output_dir", self.output_dir)))
        self.output_width.setValue(int(data.get("output_width", self.output_width.value())))
        self.output_height.setValue(int(data.get("output_height", self.output_height.value())))
        self.jpeg_quality.setValue(int(data.get("jpeg_quality", self.jpeg_quality.value())))
        self.gap_pixels.setValue(int(data.get("gap_pixels", self.gap_pixels.value())))
        self.crop_scale.setValue(int(data.get("crop_scale_percent", self.crop_scale.value())))
        self.y_offset.setValue(int(data.get("y_offset_percent", self.y_offset.value())))

        color_mode = str(data.get("color_mode", self.color_mode.currentText()))
        color_index = self.color_mode.findText(color_mode)
        if color_index >= 0:
            self.color_mode.setCurrentIndex(color_index)

        self.mirror_horizontal.setChecked(bool(data.get("mirror_horizontal", self.mirror_horizontal.isChecked())))
        self.lock_aspect.setChecked(bool(data.get("lock_aspect", self.lock_aspect.isChecked())))
        self.edit_search_area.setChecked(bool(data.get("edit_search", self.edit_search_area.isChecked())))

        search_data = data.get("search_area")
        if isinstance(search_data, dict):
            try:
                x1 = int(search_data["x"])
                y1 = int(search_data["y"])
                width = int(search_data["width"])
                height = int(search_data["height"])
            except (KeyError, TypeError, ValueError):
                self.saved_search_area = None
            else:
                self.saved_search_area = CropBox(x1, y1, width, height)
                self.current_search_area = self.saved_search_area
                self.update_search_spins(self.saved_search_area)

    def remember_search_area_from_widgets(self) -> None:
        if self.current_image is not None:
            self.search_spins_changed()
            return
        x1 = self.search_x1.value()
        y1 = self.search_y1.value()
        x2 = self.search_x2.value()
        y2 = self.search_y2.value()
        if x2 > x1 and y2 > y1:
            self.saved_search_area = CropBox(x1, y1, x2 - x1, y2 - y1)

    def args(self) -> argparse.Namespace:
        search_x = None
        search_y = None
        if self.current_search_area is not None:
            search_x = f"{self.current_search_area.x}:{self.current_search_area.x2 + 1}"
            search_y = f"{self.current_search_area.y}:{self.current_search_area.y2 + 1}"

        return argparse.Namespace(
            threshold=250,
            max_chroma=10,
            min_area=1000,
            perforation_search_x=search_x,
            perforation_search_y=search_y,
            color_mode=self.color_mode.currentText(),
            artifact_threshold=0.12,
            mirror_horizontal=self.mirror_horizontal.isChecked(),
            frame_aspect=4 / 3,
            frame_height_perf_ratio=3.5,
            crop_scale=self.crop_scale.value() / 100.0,
            inter_frame_gap_pixels=self.gap_pixels.value(),
            right_gap_perf_ratio=0.0,
            center_y_offset_perf_ratio=self.y_offset.value() / 100.0,
            no_deskew=False,
            max_deskew_degrees=8.0,
            perforation_height_width_ratio=2.25,
            disable_perforation_repair=False,
            jpeg_quality=self.jpeg_quality.value(),
        )

    def output_size(self) -> tuple[int, int]:
        return self.output_width.value(), self.output_height.value()

    def open_folder(self) -> None:
        self.stop_batch()
        folder = QFileDialog.getExistingDirectory(self, "Open image folder", str(Path.cwd()))
        if not folder:
            return
        self.input_paths = image_paths_from_folder(Path(folder), "*.jpg", r"^frame_\d+\.jpe?g$")
        self.results.clear()
        self.current_search_area = self.saved_search_area
        self.populate_frames()

    def open_image(self) -> None:
        self.stop_batch()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open JPEG image",
            str(Path.cwd()),
            "JPEG images (*.jpg *.jpeg)",
        )
        if not file_name:
            return
        self.input_paths = [Path(file_name)]
        self.results.clear()
        self.current_search_area = self.saved_search_area
        self.populate_frames()

    def choose_output_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose output folder", str(self.output_dir))
        if folder:
            self.output_dir = Path(folder)
            self.save_settings()
            self.update_previews()
            self.statusBar().showMessage(f"Output: {self.output_dir}")

    def populate_frames(self) -> None:
        self.frame_list.clear()
        for path in self.input_paths:
            item = QListWidgetItem(path.name)
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.frame_list.addItem(item)
        if self.input_paths:
            self.frame_list.setCurrentRow(0)
        self.update_batch_buttons()
        self.statusBar().showMessage(f"Loaded {len(self.input_paths)} image(s)")

    def select_frame(self, row: int) -> None:
        if row < 0 or row >= len(self.input_paths):
            return
        self.current_path = self.input_paths[row]
        self.current_image = load_image(self.current_path)
        self.canvas.set_image(self.current_image)
        self.ensure_default_search_area()

        result = self.results.get(self.current_path)
        perforation = result.get("perforation") if result else None
        crop_data = result.get("crop") if result else None
        perforation_data = perforation if isinstance(perforation, dict) else None
        self.canvas.set_perforation(perforation_data)
        if isinstance(crop_data, dict):
            self.set_current_crop(CropBox(int(crop_data["x"]), int(crop_data["y"]), int(crop_data["width"]), int(crop_data["height"])))
        else:
            self.set_current_crop(self.default_manual_crop())
        self.update_previews()

    def ensure_default_search_area(self) -> None:
        if self.current_image is None:
            self.current_search_area = None
            self.canvas.set_search_area(None)
            return
        height, width = self.current_image.shape[:2]
        if self.current_search_area is None:
            if self.saved_search_area is not None:
                self.current_search_area = clamp_crop(self.saved_search_area, width, height)
            else:
                area = parse_search_area("85%:", "20%:85%")
                if area is None:
                    self.current_search_area = None
                else:
                    x1, x2, y1, y2 = area.bounds(width, height)
                    self.current_search_area = CropBox(x1, y1, x2 - x1, y2 - y1)
        else:
            self.current_search_area = clamp_crop(self.current_search_area, width, height)
        self.canvas.set_search_area(self.current_search_area)
        self.update_search_spins(self.current_search_area)

    def default_manual_crop(self) -> CropBox | None:
        if self.current_image is None:
            return None
        height, width = self.current_image.shape[:2]
        crop_height = round(height * 0.88)
        crop_width = round(crop_height * 4 / 3)
        if crop_width > width:
            crop_width = round(width * 0.88)
            crop_height = round(crop_width / (4 / 3))
        return CropBox((width - crop_width) // 2, (height - crop_height) // 2, crop_width, crop_height)

    def set_current_crop(self, crop: CropBox | None) -> None:
        self.current_crop = crop
        self.canvas.set_crop(crop)
        self.update_crop_spins(crop)

    def update_crop_spins(self, crop: CropBox | None) -> None:
        for spin in (self.crop_x, self.crop_y, self.crop_w, self.crop_h):
            spin.blockSignals(True)
        if crop is not None:
            self.crop_x.setValue(crop.x)
            self.crop_y.setValue(crop.y)
            self.crop_w.setValue(crop.width)
            self.crop_h.setValue(crop.height)
        for spin in (self.crop_x, self.crop_y, self.crop_w, self.crop_h):
            spin.blockSignals(False)

    def update_search_spins(self, search_area: CropBox | None) -> None:
        for spin in (self.search_x1, self.search_x2, self.search_y1, self.search_y2):
            spin.blockSignals(True)
        if search_area is not None:
            self.search_x1.setValue(search_area.x)
            self.search_x2.setValue(search_area.x2 + 1)
            self.search_y1.setValue(search_area.y)
            self.search_y2.setValue(search_area.y2 + 1)
        for spin in (self.search_x1, self.search_x2, self.search_y1, self.search_y2):
            spin.blockSignals(False)

    def canvas_crop_changed(self, crop: CropBox) -> None:
        self.current_crop = crop
        self.update_crop_spins(crop)
        self.update_previews()

    def crop_spins_changed(self) -> None:
        if self.current_image is None:
            return
        height, width = self.current_image.shape[:2]
        crop = clamp_crop(
            CropBox(self.crop_x.value(), self.crop_y.value(), self.crop_w.value(), self.crop_h.value()),
            width,
            height,
        )
        self.current_crop = crop
        self.canvas.set_crop(crop)
        self.update_previews()

    def canvas_search_area_changed(self, search_area: CropBox) -> None:
        self.current_search_area = search_area
        self.saved_search_area = search_area
        self.update_search_spins(search_area)

    def search_spins_changed(self) -> None:
        if self.current_image is None:
            return
        height, width = self.current_image.shape[:2]
        x1 = self.search_x1.value()
        y1 = self.search_y1.value()
        x2 = self.search_x2.value() or width
        y2 = self.search_y2.value() or height
        search_area = clamp_crop(CropBox(x1, y1, max(1, x2 - x1), max(1, y2 - y1)), width, height)
        self.current_search_area = search_area
        self.saved_search_area = search_area
        self.canvas.set_search_area(search_area)

    def current_output_path(self, path: Path) -> Path:
        return output_path_for_image(path, self.output_dir, True)

    def current_debug_path(self, path: Path) -> Path:
        return self.output_dir / "debug" / f"{path.stem}_debug.jpg"

    def next_path_for(self, path: Path) -> Path | None:
        try:
            index = self.input_paths.index(path)
        except ValueError:
            return None
        if index + 1 < len(self.input_paths):
            return self.input_paths[index + 1]
        return None

    def auto_current(self) -> None:
        if self.current_path is None:
            return
        result = self.process_one(self.current_path)
        self.results[self.current_path] = result
        self.mark_item(self.current_path, result)
        self.select_frame(self.frame_list.currentRow())
        self.write_log()

    def start_batch(self) -> None:
        if not self.input_paths:
            return
        self.batch_index = 0
        self.batch_running = True
        self.batch_paused = False
        self.update_batch_buttons()
        self.statusBar().showMessage("Batch normalization started")
        self.batch_timer.start(0)

    def pause_batch(self) -> None:
        if not self.batch_running:
            return
        self.batch_paused = True
        self.batch_timer.stop()
        self.update_batch_buttons()
        self.write_log()
        self.statusBar().showMessage(f"Paused at {self.batch_index}/{len(self.input_paths)}")

    def resume_batch(self) -> None:
        if not self.batch_running or not self.batch_paused:
            return
        self.batch_paused = False
        self.update_batch_buttons()
        self.statusBar().showMessage(f"Resumed at {self.batch_index + 1}/{len(self.input_paths)}")
        self.batch_timer.start(0)

    def stop_batch(self) -> None:
        if not self.batch_running:
            return
        self.batch_timer.stop()
        self.batch_running = False
        self.batch_paused = False
        self.update_batch_buttons()
        self.write_log()
        self.statusBar().showMessage(f"Stopped at {self.batch_index}/{len(self.input_paths)}")

    def finish_batch(self) -> None:
        self.batch_timer.stop()
        self.batch_running = False
        self.batch_paused = False
        self.update_batch_buttons()
        self.write_log()
        self.select_frame(self.frame_list.currentRow())
        self.statusBar().showMessage("Batch normalization complete")

    def process_next_batch_frame(self) -> None:
        if not self.batch_running or self.batch_paused:
            return
        if self.batch_index >= len(self.input_paths):
            self.finish_batch()
            return

        path = self.input_paths[self.batch_index]
        result = self.process_one(path)
        self.results[path] = result
        self.mark_item(path, result)
        self.batch_index += 1

        if path == self.current_path:
            self.select_frame(self.frame_list.currentRow())

        self.statusBar().showMessage(f"Processed {self.batch_index}/{len(self.input_paths)}")
        self.batch_timer.start(0)

    def update_batch_buttons(self) -> None:
        has_inputs = bool(self.input_paths)
        self.start_button.setEnabled(has_inputs and not self.batch_running)
        self.pause_button.setEnabled(self.batch_running and not self.batch_paused)
        self.resume_button.setEnabled(self.batch_running and self.batch_paused)
        self.stop_button.setEnabled(self.batch_running)
        self.open_folder_button.setEnabled(not self.batch_running)
        self.open_image_button.setEnabled(not self.batch_running)
        self.choose_output_button.setEnabled(not self.batch_running)
        self.detect_current_button.setEnabled(has_inputs and not self.batch_running)

    def process_one(self, path: Path) -> dict[str, Any]:
        output_path = self.current_output_path(path)
        annotation_path = self.current_debug_path(path)
        try:
            result = process_image(
                path,
                output_path,
                annotation_path,
                self.output_size(),
                self.args(),
                next_image_path=self.next_path_for(path),
            )
            result["status"] = "ok"
            return result
        except Exception as error:
            image = load_image(path)
            return {
                "image": str(path),
                "image_width": image.shape[1],
                "image_height": image.shape[0],
                "perforation": None,
                "crop": None,
                "vertical_stitch": None,
                "vertical_stitch_pixels": 0,
                "deskew_degrees": 0.0,
                "normalized_output": str(output_path),
                "normalized_width": self.output_width.value(),
                "normalized_height": self.output_height.value(),
                "jpeg_quality": self.jpeg_quality.value(),
                "color_mode": self.color_mode.currentText(),
                "mirror_horizontal": self.mirror_horizontal.isChecked(),
                "candidate_count": 0,
                "status": "needs_manual",
                "fallback_reason": str(error),
            }

    def mark_item(self, path: Path, result: dict[str, Any]) -> None:
        for row in range(self.frame_list.count()):
            item = self.frame_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == path:
                status = result.get("status", "")
                prefix = "OK" if status == "ok" else "MANUAL" if status == "needs_manual" else str(status).upper()
                item.setText(f"{prefix}  {path.name}")
                return

    def save_manual(self) -> None:
        if self.current_path is None or self.current_image is None or self.current_crop is None:
            return
        output_path = self.current_output_path(self.current_path)
        color_mode, artifact_fraction, stitch = crop_and_normalize(
            self.current_image,
            self.current_path,
            self.current_crop,
            output_path,
            self.output_size(),
            self.color_mode.currentText(),
            0.12,
            self.mirror_horizontal.isChecked(),
            self.jpeg_quality.value(),
            None,
            None,
            self.gap_pixels.value(),
        )
        result = {
            "image": str(self.current_path),
            "image_width": self.current_image.shape[1],
            "image_height": self.current_image.shape[0],
            "perforation": None,
            "crop": self.current_crop.to_log_dict(),
            "vertical_stitch": stitch.to_log_dict(),
            "vertical_stitch_pixels": stitch.pixels,
            "deskew_degrees": 0.0,
            "normalized_output": str(output_path),
            "normalized_width": self.output_width.value(),
            "normalized_height": self.output_height.value(),
            "jpeg_quality": self.jpeg_quality.value(),
            "color_mode": color_mode,
            "mirror_horizontal": self.mirror_horizontal.isChecked(),
            "cyan_artifact_fraction": round(artifact_fraction, 4),
            "candidate_count": 0,
            "status": "manual",
        }
        self.results[self.current_path] = result
        self.mark_item(self.current_path, result)
        self.update_previews()
        self.write_log()
        self.statusBar().showMessage(f"Saved {output_path}")

    def set_image_preview(self, label: ImagePreview, image_path: Path, missing_text: str, unreadable_text: str) -> None:
        if not image_path.exists():
            label.set_message(missing_text)
            return
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            label.set_message(unreadable_text)
            return
        label.set_cv_image(image)

    def update_debug_preview(self) -> None:
        if self.current_path is None:
            self.debug_preview.set_message("Debug preview")
            return
        self.set_image_preview(
            self.debug_preview,
            self.current_debug_path(self.current_path),
            "No debug yet",
            "Could not read debug",
        )

    def update_normalized_preview(self) -> None:
        if self.current_path is None:
            self.normalized_preview.set_message("Normalized preview")
            return
        output_path = self.current_output_path(self.current_path)
        self.set_image_preview(
            self.normalized_preview,
            output_path,
            "No output yet",
            "Could not read output",
        )

    def update_previews(self) -> None:
        self.update_debug_preview()
        self.update_normalized_preview()

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self.update_previews()

    def write_log(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        frames = [self.results[path] for path in self.input_paths if path in self.results]
        data = {
            "input": str(self.input_paths[0].parent) if self.input_paths else "",
            "output": str(self.output_dir),
            "count": len(self.input_paths),
            "processed": len(frames),
            "frames": frames,
        }
        (self.output_dir / "positions.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    def closeEvent(self, event: Any) -> None:
        self.save_settings()
        if self.results:
            self.write_log()
        event.accept()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
