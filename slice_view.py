"""スライス画像 1 枚を表示するウィジェットと、スライダー付きパネル。"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (QGridLayout, QGroupBox, QLabel, QSlider,
                               QVBoxLayout, QWidget)


def apply_window(img: np.ndarray, wl: float, ww: float) -> np.ndarray:
    """HU 画像に WL/WW を適用して 0-255 の uint8 画像にする。"""
    ww = max(ww, 1.0)
    lo = wl - ww / 2.0
    out = (img.astype(np.float32) - lo) * (255.0 / ww)
    return np.clip(out, 0, 255).astype(np.uint8)


class SliceView(QWidget):
    """画像 + 他 2 断面の位置を示す線を描画する。

    画像のピクセル間隔(mm)を考慮して縦横比を保って表示する。
    """

    picked = Signal(int, int)   # クリック/ドラッグ位置 (列, 行) = 画像ピクセル座標
    wheel = Signal(int)         # ホイール(+1 / -1)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.setMouseTracking(False)
        self._buf: Optional[np.ndarray] = None
        self._qimg: Optional[QImage] = None
        self._col_mm = 1.0
        self._row_mm = 1.0
        self._hline: Optional[float] = None   # 水平線の行位置(ピクセル index)
        self._vline: Optional[float] = None   # 垂直線の列位置
        self._hcolor = QColor("white")
        self._vcolor = QColor("white")

    # ---- 外部から設定 ----
    def set_image(self, gray_u8: np.ndarray, col_mm: float, row_mm: float):
        self._buf = np.ascontiguousarray(gray_u8)  # QImage が参照するので保持
        h, w = self._buf.shape
        self._qimg = QImage(self._buf.data, w, h, w, QImage.Format_Grayscale8)
        self._col_mm, self._row_mm = col_mm, row_mm
        self.update()

    def set_lines(self, hline, hcolor, vline, vcolor):
        self._hline, self._vline = hline, vline
        self._hcolor, self._vcolor = QColor(hcolor), QColor(vcolor)
        self.update()

    # ---- 描画 ----
    def _image_rect(self) -> Optional[QRectF]:
        if self._qimg is None:
            return None
        w, h = self._qimg.width(), self._qimg.height()
        pw, ph = w * self._col_mm, h * self._row_mm
        scale = min(self.width() / pw, self.height() / ph)
        rw, rh = pw * scale, ph * scale
        return QRectF((self.width() - rw) / 2, (self.height() - rh) / 2, rw, rh)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.black)
        rect = self._image_rect()
        if rect is None:
            return
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        p.drawImage(rect, self._qimg)
        w, h = self._qimg.width(), self._qimg.height()
        if self._hline is not None:
            y = rect.top() + (self._hline + 0.5) / h * rect.height()
            p.setPen(QPen(self._hcolor, 2))
            p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        if self._vline is not None:
            x = rect.left() + (self._vline + 0.5) / w * rect.width()
            p.setPen(QPen(self._vcolor, 2))
            p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))

    # ---- マウス ----
    def _emit_pick(self, pos):
        rect = self._image_rect()
        if rect is None or rect.width() <= 0:
            return
        w, h = self._qimg.width(), self._qimg.height()
        col = int((pos.x() - rect.left()) / rect.width() * w)
        row = int((pos.y() - rect.top()) / rect.height() * h)
        self.picked.emit(min(max(col, 0), w - 1), min(max(row, 0), h - 1))

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._emit_pick(e.position())

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton:
            self._emit_pick(e.position())

    def wheelEvent(self, e):
        d = e.angleDelta().y()
        if d:
            self.wheel.emit(1 if d > 0 else -1)


class LabeledSlider(QWidget):
    """名前 + スライダー + 現在値ラベル。"""

    valueChanged = Signal(int)

    def __init__(self, name: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        lay = QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._name = QLabel(name)
        self._name.setMinimumWidth(48)
        self.slider = QSlider(Qt.Horizontal)
        self._val = QLabel("0")
        self._val.setMinimumWidth(70)
        self._val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay.addWidget(self._name, 0, 0)
        lay.addWidget(self.slider, 0, 1)
        lay.addWidget(self._val, 0, 2)
        lay.setColumnStretch(1, 1)
        self.slider.valueChanged.connect(self._on_changed)
        self._fmt = "{}"

    def _on_changed(self, v: int):
        self._val.setText(self._fmt.format(v))
        self.valueChanged.emit(v)

    def configure(self, lo: int, hi: int, value: int, fmt: str = "{}"):
        self._fmt = fmt
        self.slider.blockSignals(True)
        self.slider.setRange(lo, hi)
        self.slider.setValue(value)
        self.slider.blockSignals(False)
        self._val.setText(fmt.format(self.slider.value()))

    def set_value(self, v: int):
        """シグナルを出さずに値だけ更新(他ビューからの連動用)。"""
        self.slider.blockSignals(True)
        self.slider.setValue(v)
        self.slider.blockSignals(False)
        self._val.setText(self._fmt.format(self.slider.value()))

    def value(self) -> int:
        return self.slider.value()


class ViewPanel(QGroupBox):
    """断面 1 つ分: 画像 + スライス位置/WL/WW スライダー。"""

    sliceChanged = Signal(int)
    windowChanged = Signal()

    def __init__(self, title: str, color: str, parent: Optional[QWidget] = None):
        super().__init__(title, parent)
        self.setStyleSheet(
            f"QGroupBox {{ font-weight: bold; border: 2px solid {color}; "
            f"border-radius: 4px; margin-top: 10px; }}"
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 8px; color: {color}; }}")
        self.view = SliceView()
        self.slice_slider = LabeledSlider("Slice")
        self.wl_slider = LabeledSlider("WL")
        self.ww_slider = LabeledSlider("WW")
        lay = QVBoxLayout(self)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.slice_slider)
        lay.addWidget(self.wl_slider)
        lay.addWidget(self.ww_slider)
        self.slice_slider.valueChanged.connect(self.sliceChanged)
        self.wl_slider.valueChanged.connect(lambda _v: self.windowChanged.emit())
        self.ww_slider.valueChanged.connect(lambda _v: self.windowChanged.emit())

    @property
    def wl(self) -> float:
        return float(self.wl_slider.value())

    @property
    def ww(self) -> float:
        return float(self.ww_slider.value())

    def set_window(self, wl: float, ww: float):
        self.wl_slider.set_value(int(round(wl)))
        self.ww_slider.set_value(int(round(ww)))
