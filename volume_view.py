"""観点7: 肺領域のボリュームレンダリング表示ウィンドウ(VTK 使用)。

CT 全体のうち、肺マスクの内側だけを描画する。
  - 「肺領域の CT」: マスク内の CT 値に色と不透明度を割り当てて描画(血管・結節が見える)
  - 「肺マスクのみ」: マスクを平滑化して単色で描画(肺の形状を見る)
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import vtk
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QGridLayout, QGroupBox, QHBoxLayout,
                               QLabel, QMainWindow, QPushButton, QSlider,
                               QComboBox, QVBoxLayout, QWidget)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util.numpy_support import numpy_to_vtk

MODE_CT = "肺領域の CT(CT 値で着色)"
MODE_MASK = "肺マスクのみ(単色)"

# (HU, R, G, B) / (HU, 不透明度)
CT_COLORS = [(-1024, 0.0, 0.0, 0.0), (-900, 0.35, 0.60, 1.00), (-700, 0.55, 0.80, 1.00),
             (-400, 1.00, 0.75, 0.60), (0, 1.0, 0.85, 0.75), (300, 1.0, 1.0, 1.0)]
CT_OPACITY = [(-1024, 0.0), (-950, 0.0), (-880, 0.10), (-650, 0.18),
              (-300, 0.45), (-100, 0.80), (100, 0.95)]


class VolumeWindow(QMainWindow):
    def __init__(self, ct: np.ndarray, mask: np.ndarray,
                 spacing: Tuple[float, float, float], parent=None):
        super().__init__(parent)
        self.setWindowTitle("肺領域 3D ボリュームレンダリング")
        self.resize(1000, 760)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self._ct, self._mask, self._sp = ct, mask, spacing
        self._first = True

        # マスクの z 方向が CT と逆かどうかを自動判定(肺は空気なので CT 値が低いほうが正しい)
        mean_n = float(ct[mask].mean())
        mean_f = float(ct[mask[::-1]].mean())
        auto_flip = mean_f < mean_n - 1.0

        # ---- VTK ----
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.05, 0.05, 0.10)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.iren = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.mapper = vtk.vtkSmartVolumeMapper()
        self.mapper.SetBlendModeToComposite()
        self.prop = vtk.vtkVolumeProperty()
        self.prop.SetInterpolationTypeToLinear()
        self.prop.ShadeOn()
        self.prop.SetAmbient(0.25)
        self.prop.SetDiffuse(0.9)
        self.prop.SetSpecular(0.2)
        self.volume = vtk.vtkVolume()
        self.volume.SetMapper(self.mapper)
        self.volume.SetProperty(self.prop)
        self.renderer.AddVolume(self.volume)
        self._image = None

        # ---- 操作パネル ----
        panel = QGroupBox("表示設定")
        panel.setFixedWidth(250)
        pl = QVBoxLayout(panel)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([MODE_CT, MODE_MASK])
        pl.addWidget(QLabel("表示モード"))
        pl.addWidget(self.mode_combo)
        self.opacity_label = QLabel("不透明度: 100 %")
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(10, 300)
        self.opacity_slider.setValue(100)
        pl.addWidget(self.opacity_label)
        pl.addWidget(self.opacity_slider)
        self.flip_cb = QCheckBox("マスクの z 方向を反転")
        self.flip_cb.setChecked(auto_flip)
        pl.addWidget(self.flip_cb)
        pl.addWidget(QLabel("視点"))
        grid = QGridLayout()
        for i, (label, key) in enumerate([("正面", "anterior"), ("背面", "posterior"),
                                          ("右側", "right"), ("左側", "left"),
                                          ("上から", "superior"), ("下から", "inferior")]):
            b = QPushButton(label)
            b.clicked.connect(lambda _c=False, k=key: self.set_view(k))
            grid.addWidget(b, i // 2, i % 2)
        pl.addLayout(grid)
        self.info = QLabel()
        self.info.setWordWrap(True)
        pl.addSpacing(8)
        pl.addWidget(self.info)
        pl.addStretch(1)
        help_label = QLabel("左ドラッグ: 回転\n右ドラッグ / ホイール: 拡大縮小\n"
                            "中ドラッグ(Shift+左): 平行移動")
        help_label.setStyleSheet("color: gray;")
        pl.addWidget(help_label)

        central = QWidget()
        lay = QHBoxLayout(central)
        lay.addWidget(self.vtk_widget, 1)
        lay.addWidget(panel)
        self.setCentralWidget(central)

        self.mode_combo.currentIndexChanged.connect(self._rebuild)
        self.flip_cb.toggled.connect(self._rebuild)
        self.opacity_slider.valueChanged.connect(self._on_opacity)
        self._rebuild()

    # ---------- ボリューム作成 ----------
    def _to_vtk_image(self, arr: np.ndarray, vtk_type: int, origin_vox) -> vtk.vtkImageData:
        nz, ny, nx = arr.shape
        sx, sy, sz = self._sp
        img = vtk.vtkImageData()
        img.SetDimensions(nx, ny, nz)
        img.SetSpacing(sx, sy, sz)
        img.SetOrigin(origin_vox[0] * sx, origin_vox[1] * sy, origin_vox[2] * sz)
        img.GetPointData().SetScalars(
            numpy_to_vtk(np.ascontiguousarray(arr).ravel(), deep=True, array_type=vtk_type))
        return img

    def _rebuild(self, *_args):
        mask = self._mask[::-1] if self.flip_cb.isChecked() else self._mask
        zs = np.flatnonzero(mask.any(axis=(1, 2)))
        ys = np.flatnonzero(mask.any(axis=(0, 2)))
        xs = np.flatnonzero(mask.any(axis=(0, 1)))
        if zs.size == 0:
            return
        nz, ny, nx = mask.shape
        mg = 3  # 余白
        z0, z1 = max(zs[0] - mg, 0), min(zs[-1] + mg + 1, nz)
        y0, y1 = max(ys[0] - mg, 0), min(ys[-1] + mg + 1, ny)
        x0, x1 = max(xs[0] - mg, 0), min(xs[-1] + mg + 1, nx)
        m = mask[z0:z1, y0:y1, x0:x1]
        origin = (x0, y0, z0)

        if self.mode_combo.currentText() == MODE_CT:
            ct = self._ct[z0:z1, y0:y1, x0:x1]
            arr = np.where(m, ct, -1024).astype(np.int16)
            self._image = self._to_vtk_image(arr, vtk.VTK_SHORT, origin)
        else:
            arr = m.astype(np.float32) * 255.0
            img = self._to_vtk_image(arr, vtk.VTK_FLOAT, origin)
            g = vtk.vtkImageGaussianSmooth()
            g.SetInputData(img)
            g.SetDimensionality(3)
            sx, sy, sz = self._sp  # 平滑化の強さを mm 基準にそろえる(非等方ボクセル対策)
            g.SetStandardDeviations(2.5 / sx, 2.5 / sy, 2.5 / sz)
            g.Update()
            self._image = vtk.vtkImageData()
            self._image.DeepCopy(g.GetOutput())
        self.mapper.SetInputData(self._image)
        self._apply_transfer()

        sx, sy, sz = self._sp
        vol_ml = float(m.sum()) * sx * sy * sz / 1000.0
        self.info.setText(f"肺領域: {int(m.sum()):,} ボクセル\n体積: {vol_ml:,.0f} mL\n"
                          f"描画範囲: {x1 - x0}×{y1 - y0}×{z1 - z0}")
        if self._first:
            self._first = False
            self.set_view("anterior")
        else:
            self.vtk_widget.GetRenderWindow().Render()

    def _apply_transfer(self):
        scale = self.opacity_slider.value() / 100.0
        ctf = vtk.vtkColorTransferFunction()
        otf = vtk.vtkPiecewiseFunction()
        if self.mode_combo.currentText() == MODE_CT:
            for h, r, g, b in CT_COLORS:
                ctf.AddRGBPoint(h, r, g, b)
            for h, o in CT_OPACITY:
                otf.AddPoint(h, min(1.0, o * scale))
            self.prop.SetScalarOpacityUnitDistance(10.0 * min(self._sp))
        else:
            ctf.AddRGBPoint(0, 0.9, 0.55, 0.45)
            ctf.AddRGBPoint(255, 0.9, 0.55, 0.45)
            otf.AddPoint(0, 0.0)
            otf.AddPoint(90, 0.0)
            otf.AddPoint(160, min(1.0, 0.9 * scale))
            otf.AddPoint(255, min(1.0, 0.9 * scale))
            self.prop.SetScalarOpacityUnitDistance(min(self._sp))
        self.prop.SetColor(ctf)
        self.prop.SetScalarOpacity(otf)

    def _on_opacity(self, v: int):
        self.opacity_label.setText(f"不透明度: {v} %")
        self._apply_transfer()
        self.vtk_widget.GetRenderWindow().Render()

    # ---------- 視点 ----------
    def set_view(self, key: str):
        """患者座標系 (LPS: +x=左, +y=後, +z=頭) を基準にカメラを置く。"""
        if self._image is None:
            return
        b = self._image.GetBounds()
        c = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
        d = max(b[1] - b[0], b[3] - b[2], b[5] - b[4]) * 2.6
        dirs = {  # (カメラ位置の方向, 上方向)
            "anterior": ((0, -1, 0), (0, 0, 1)), "posterior": ((0, 1, 0), (0, 0, 1)),
            "left": ((1, 0, 0), (0, 0, 1)), "right": ((-1, 0, 0), (0, 0, 1)),
            "superior": ((0, 0, 1), (0, -1, 0)), "inferior": ((0, 0, -1), (0, 1, 0)),
        }
        pos, up = dirs[key]
        cam = self.renderer.GetActiveCamera()
        cam.SetFocalPoint(*c)
        cam.SetPosition(*(c + d * np.array(pos, dtype=float)))
        cam.SetViewUp(*up)
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

    def closeEvent(self, e):
        try:
            self.vtk_widget.Finalize()
        except Exception:
            pass
        super().closeEvent(e)