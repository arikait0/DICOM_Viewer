"""DICOM Viewer (Axial / Coronal / Sagittal) - 最終課題 観点1〜7。

起動:  python main.py
"""
from __future__ import annotations

import sys
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QApplication, QFileDialog, QGridLayout,
                               QInputDialog, QLabel, QMainWindow, QMessageBox,
                               QProgressDialog, QWidget)

import dicom_loader as dl
import mask_loader as ml
from slice_view import ViewPanel, apply_window

# 各断面を表す色(他ビューに描かれる線はこの色)
COLOR_AXIAL = "#ff5050"      # 赤
COLOR_CORONAL = "#40e060"    # 緑
COLOR_SAGITTAL = "#40b0ff"   # 青

WINDOW_PRESETS = {
    "軟部組織 (WL 40 / WW 400)": (40, 400),
    "肺野 (WL -600 / WW 1500)": (-600, 1500),
    "骨 (WL 300 / WW 1500)": (300, 1500),
}


class MainWindow(QMainWindow):
  def __init__(self):
    super().__init__()
    self.setWindowTitle("DICOM Viewer")
    self.resize(1400, 520)
    self.volume: Optional[dl.Volume] = None
    self.pos = [0, 0, 0]  # 現在位置 (x, y, z) ボクセル index

    self.axial = ViewPanel("Axial", COLOR_AXIAL)
    self.coronal = ViewPanel("Coronal", COLOR_CORONAL)
    self.sagittal = ViewPanel("Sagittal", COLOR_SAGITTAL)
    central = QWidget()
    lay = QGridLayout(central)
    for i, p in enumerate((self.axial, self.coronal, self.sagittal)):
      lay.addWidget(p, 0, i)
      lay.setColumnStretch(i, 1)
    self.setCentralWidget(central)

    self.info_label = QLabel("ファイル > DICOM フォルダを開く から読み込んでください")
    self.statusBar().addWidget(self.info_label)

    self._build_menu()
    self._connect_signals()
    self._set_enabled(False)

  # ---------------- メニュー ----------------
  def _build_menu(self):
    mb = self.menuBar()
    m_file = mb.addMenu("ファイル(&F)")
    a_folder = QAction("DICOM フォルダを開く...", self)
    a_folder.setShortcut(QKeySequence.Open)
    a_folder.triggered.connect(self.open_folder)
    a_files = QAction("DICOM ファイルを選択して開く...", self)
    a_files.triggered.connect(self.open_files)
    a_quit = QAction("終了", self)
    a_quit.setShortcut(QKeySequence.Quit)
    a_quit.triggered.connect(self.close)
    m_file.addAction(a_folder)
    m_file.addAction(a_files)
    m_file.addSeparator()
    m_file.addAction(a_quit)

    self.m_view = mb.addMenu("表示(&V)")
    for name, (wl, ww) in WINDOW_PRESETS.items():
      a = QAction(f"全ビューを {name} にする", self)
      a.triggered.connect(lambda _c=False, wl=wl,
                          ww=ww: self.apply_preset(wl, ww))
      self.m_view.addAction(a)
    self.m_view.addSeparator()
    a_center = QAction("スライス位置を中央に戻す", self)
    a_center.triggered.connect(self.center_position)
    self.m_view.addAction(a_center)

    # 観点7: 肺領域の 3D ボリュームレンダリング
    self.m_3d = mb.addMenu("3D表示(&3)")
    a_mfile = QAction("肺マスク TIFF ファイルを選んで 3D 表示...", self)
    a_mfile.triggered.connect(self.open_mask_file)
    a_mdir = QAction("肺マスク TIFF フォルダ(1スライス1ファイル)を選んで 3D 表示...", self)
    a_mdir.triggered.connect(self.open_mask_folder)
    self.m_3d.addAction(a_mfile)
    self.m_3d.addAction(a_mdir)

  # ---------------- シグナル接続 ----------------
  def _connect_signals(self):
    self.axial.sliceChanged.connect(lambda v: self.set_pos(z=v))
    self.coronal.sliceChanged.connect(lambda v: self.set_pos(y=v))
    self.sagittal.sliceChanged.connect(lambda v: self.set_pos(x=v))
    for p in (self.axial, self.coronal, self.sagittal):
      p.windowChanged.connect(self.refresh_images)
    # 画像クリック/ドラッグで位置指定
    self.axial.view.picked.connect(lambda c, r: self.set_pos(x=c, y=r))
    self.coronal.view.picked.connect(
        lambda c, r: self.set_pos(x=c, z=self._nz - 1 - r))
    self.sagittal.view.picked.connect(
        lambda c, r: self.set_pos(y=c, z=self._nz - 1 - r))
    # ホイールでスライス送り
    self.axial.view.wheel.connect(lambda d: self.set_pos(z=self.pos[2] + d))
    self.coronal.view.wheel.connect(
        lambda d: self.set_pos(y=self.pos[1] + d))
    self.sagittal.view.wheel.connect(
        lambda d: self.set_pos(x=self.pos[0] + d))

  def _set_enabled(self, on: bool):
    for p in (self.axial, self.coronal, self.sagittal):
      p.setEnabled(on)
    self.m_view.setEnabled(on)
    self.m_3d.setEnabled(on)

  @property
  def _nz(self) -> int:
    return self.volume.data.shape[0]

  # ---------------- 読み込み ----------------
  def open_folder(self):
    folder = QFileDialog.getExistingDirectory(self, "DICOM フォルダを選択")
    if folder:
      self._load(dl.collect_paths(folder))

  def open_files(self):
    paths, _ = QFileDialog.getOpenFileNames(
        self, "DICOM ファイルを選択 (複数可)", "", "DICOM (*.dcm *.DCM *.ima *.IMA);;All files (*)")
    if paths:
      self._load(paths)

  def _progress(self, text: str, total: int) -> QProgressDialog:
    dlg = QProgressDialog(text, None, 0, total, self)
    dlg.setWindowModality(Qt.WindowModal)
    dlg.setMinimumDuration(0)
    dlg.setValue(0)
    return dlg

  def _load(self, paths):
    if not paths:
      return
    dlg = self._progress("DICOM を検索中...", len(paths))

    def cb(i, total):
      dlg.setMaximum(max(total, 1))
      dlg.setValue(i)
      QApplication.processEvents()

    series = dl.find_series(paths, cb)
    dlg.close()
    if not series:
      QMessageBox.warning(self, "読み込みエラー", "画像を含む DICOM ファイルが見つかりませんでした。")
      return
    chosen = series[0]
    if len(series) > 1:
      labels = [s.label for s in series]
      item, ok = QInputDialog.getItem(self, "シリーズ選択",
                                      "複数のシリーズがあります。表示するシリーズを選択:",
                                      labels, 0, False)
      if not ok:
        return
      chosen = series[labels.index(item)]

    dlg = self._progress("読み込み中...", len(chosen.paths))

    def cb2(i, total):
      dlg.setMaximum(max(total, 1))
      dlg.setValue(i)
      QApplication.processEvents()

    try:
      vol = dl.load_series(chosen, cb2)
    except dl.DicomLoadError as e:
      dlg.close()
      QMessageBox.critical(self, "読み込みエラー", str(e))
      return
    dlg.close()
    self.set_volume(vol)

  def set_volume(self, vol: dl.Volume):
    self.volume = vol
    nx, ny, nz = vol.shape_xyz
    vmin, vmax = int(vol.data.min()), int(vol.data.max())
    wl_lo, wl_hi = vmin, max(vmax, vmin + 1)
    ww_hi = max(vmax - vmin, 2)
    wl = int(np.clip(round(vol.window_center), wl_lo, wl_hi))
    ww = int(np.clip(round(vol.window_width), 1, ww_hi))
    for p, n in ((self.axial, nz), (self.coronal, ny), (self.sagittal, nx)):
      p.slice_slider.configure(0, n - 1, n // 2, "{}")
      p.wl_slider.configure(wl_lo, wl_hi, wl, "{} HU")
      p.ww_slider.configure(1, ww_hi, ww, "{}")
    self._wl_range = (wl_lo, wl_hi, ww_hi)
    self.pos = [nx // 2, ny // 2, nz // 2]
    self._set_enabled(True)
    self.setWindowTitle(f"DICOM Viewer - {vol.description or 'series'}")
    self.refresh_all()

  # ---------------- 3D 表示(観点7) ----------------
  def open_mask_file(self):
    path, _ = QFileDialog.getOpenFileName(
        self, "肺マスク TIFF を選択", "", "TIFF (*.tif *.tiff);;All files (*)")
    if path:
      self._show_3d(path)

  def open_mask_folder(self):
    path = QFileDialog.getExistingDirectory(self, "肺マスク TIFF が入ったフォルダを選択")
    if path:
      self._show_3d(path)

  def _show_3d(self, path: str):
    if self.volume is None:
      return
    QApplication.setOverrideCursor(Qt.WaitCursor)
    try:
      mask = ml.load_mask(path)
    except ml.MaskLoadError as e:
      QApplication.restoreOverrideCursor()
      QMessageBox.critical(self, "マスク読み込みエラー", str(e))
      return
    QApplication.restoreOverrideCursor()
    ct_shape = self.volume.data.shape
    if mask.shape != ct_shape:
      QMessageBox.critical(
          self, "サイズ不一致",
          f"マスクの大きさ (z, y, x) = {mask.shape}\n"
          f"CT の大きさ   (z, y, x) = {ct_shape}\n\n"
          "マスクの元になった CT と同じシリーズを開いてください。")
      return
    if not mask.any():
      QMessageBox.warning(self, "マスクが空です", "マスクに 0 以外の値がありません。")
      return
    try:
      from volume_view import VolumeWindow  # VTK が無くても 1〜6 は動くよう遅延 import
    except ImportError as e:
      QMessageBox.critical(self, "VTK が見つかりません",
                           f"pip install -r requirements.txt を実行してください。\n{e}")
      return
    QApplication.setOverrideCursor(Qt.WaitCursor)
    try:
      self._vol_win = VolumeWindow(
          self.volume.data, mask, self.volume.spacing, self)
      self._vol_win.show()
    finally:
      QApplication.restoreOverrideCursor()

  # ---------------- 操作 ----------------
  def set_pos(self, x=None, y=None, z=None):
    if self.volume is None:
      return
    nx, ny, nz = self.volume.shape_xyz
    if x is not None:
      self.pos[0] = int(np.clip(x, 0, nx - 1))
    if y is not None:
      self.pos[1] = int(np.clip(y, 0, ny - 1))
    if z is not None:
      self.pos[2] = int(np.clip(z, 0, nz - 1))
    self.refresh_all()

  def center_position(self):
    if self.volume:
      nx, ny, nz = self.volume.shape_xyz
      self.set_pos(nx // 2, ny // 2, nz // 2)

  def apply_preset(self, wl, ww):
    if not self.volume:
      return
    for p in (self.axial, self.coronal, self.sagittal):
      p.set_window(wl, ww)
    self.refresh_images()

  # ---------------- 描画更新 ----------------
  def refresh_all(self):
    """位置スライダーを同期し、画像と線を更新する(観点6: 線がスライダーに連動)。"""
    x, y, z = self.pos
    self.axial.slice_slider.set_value(z)
    self.coronal.slice_slider.set_value(y)
    self.sagittal.slice_slider.set_value(x)
    self.refresh_images()

  def refresh_images(self):
    if self.volume is None:
      return
    vol = self.volume.data
    sx, sy, sz = self.volume.spacing
    nz = vol.shape[0]
    x, y, z = self.pos
    zrow = nz - 1 - z  # 冠状/矢状は上(頭側)が上になるよう反転表示

    # Axial: 行=y, 列=x  -> 他の線: Coronal(水平 y) / Sagittal(垂直 x)
    img = apply_window(vol[z], self.axial.wl, self.axial.ww)
    self.axial.view.set_image(img, sx, sy)
    self.axial.view.set_lines(y, COLOR_CORONAL, x, COLOR_SAGITTAL)

    # Coronal: 行=z(反転), 列=x -> Axial(水平 z) / Sagittal(垂直 x)
    img = apply_window(
        np.flipud(vol[:, y, :]), self.coronal.wl, self.coronal.ww)
    self.coronal.view.set_image(img, sx, sz)
    self.coronal.view.set_lines(zrow, COLOR_AXIAL, x, COLOR_SAGITTAL)

    # Sagittal: 行=z(反転), 列=y -> Axial(水平 z) / Coronal(垂直 y)
    img = apply_window(
        np.flipud(vol[:, :, x]), self.sagittal.wl, self.sagittal.ww)
    self.sagittal.view.set_image(img, sy, sz)
    self.sagittal.view.set_lines(zrow, COLOR_AXIAL, y, COLOR_CORONAL)

    nx, ny, nz = self.volume.shape_xyz
    self.info_label.setText(
        f"位置 (x, y, z) = ({x}, {y}, {z})   値 = {int(vol[z, y, x])} HU   "
        f"サイズ {nx}×{ny}×{nz}   間隔 {sx:.2f}×{sy:.2f}×{sz:.2f} mm")


def main():
  app = QApplication(sys.argv)
  win = MainWindow()
  win.show()
  # 任意: python main.py <フォルダ> で起動時に読み込み(動作確認用)
  if len(sys.argv) > 1:
    win._load(dl.collect_paths(sys.argv[1]))
  sys.exit(app.exec())


if __name__ == "__main__":
  main()
