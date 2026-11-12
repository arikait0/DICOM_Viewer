"""DICOM 読み込みモジュール(GUI に依存しない)。

- フォルダ(再帰)またはファイル群から DICOM シリーズを検出
- ImagePositionPatient / ImageOrientationPatient でスライスを並べ替え
- RescaleSlope / RescaleIntercept を適用して HU 値の 3D 配列 (z, y, x) を作成
"""
from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pydicom

ProgressCb = Optional[Callable[[int, int], None]]


class DicomLoadError(Exception):
  """読み込み失敗時に GUI へ伝えるための例外。"""


@dataclass
class SeriesInfo:
  uid: str
  description: str
  paths: List[str]

  @property
  def label(self) -> str:
    return f"{self.description or '(no description)'}  [{len(self.paths)} 枚]"


@dataclass
class Volume:
  data: np.ndarray                      # (z, y, x) int16, HU
  spacing: Tuple[float, float, float]   # (sx, sy, sz) [mm]
  description: str
  window_center: float
  window_width: float

  @property
  def shape_xyz(self) -> Tuple[int, int, int]:
    z, y, x = self.data.shape
    return x, y, z


def collect_paths(folder: str) -> List[str]:
  """フォルダ以下の全ファイルパスを再帰的に集める。"""
  paths = []
  for root, _dirs, files in os.walk(folder):
    for f in files:
      paths.append(os.path.join(root, f))
  return sorted(paths)


def find_series(paths: Sequence[str], progress: ProgressCb = None) -> List[SeriesInfo]:
  """パス一覧から画像を持つ DICOM を探し、SeriesInstanceUID ごとにまとめる。"""
  groups: Dict[str, List[str]] = defaultdict(list)
  desc: Dict[str, str] = {}
  total = len(paths)
  for i, p in enumerate(paths):
    if progress and i % 20 == 0:
      progress(i, total)
    try:
      ds = pydicom.dcmread(p, stop_before_pixels=True)
    except Exception:
      continue  # DICOM ではないファイルは無視
    if "Rows" not in ds or "Columns" not in ds:
      continue
    uid = str(getattr(ds, "SeriesInstanceUID", "unknown"))
    groups[uid].append(p)
    desc.setdefault(uid, str(getattr(ds, "SeriesDescription", "")))
  if progress:
    progress(total, total)
  series = [SeriesInfo(uid, desc[uid], ps) for uid, ps in groups.items()]
  series.sort(key=lambda s: -len(s.paths))
  return series


def _slice_sort_key(ds) -> float:
  ipp = getattr(ds, "ImagePositionPatient", None)
  iop = getattr(ds, "ImageOrientationPatient", None)
  if ipp is not None and iop is not None:
    row = np.array(iop[:3], dtype=float)
    col = np.array(iop[3:], dtype=float)
    normal = np.cross(row, col)
    return float(np.dot(normal, np.array(ipp, dtype=float)))
  if ipp is not None:
    return float(ipp[2])
  return float(getattr(ds, "InstanceNumber", 0))


def load_series(series: SeriesInfo, progress: ProgressCb = None) -> Volume:
  """1 シリーズを読み込み Volume を返す。"""
  items = []
  total = len(series.paths)
  for i, p in enumerate(series.paths):
    if progress:
      progress(i, total)
    try:
      ds = pydicom.dcmread(p)
      _ = ds.pixel_array  # ここで圧縮形式などのエラーを検出
    except Exception as e:
      raise DicomLoadError(
          f"画像を読み込めませんでした: {os.path.basename(p)}\n{e}\n"
          "(圧縮 DICOM の場合は README の注意事項を参照)"
      )
    items.append((_slice_sort_key(ds), ds))
  items.sort(key=lambda t: t[0])
  if not items:
    raise DicomLoadError("画像が見つかりませんでした。")

  shapes = {(ds.Rows, ds.Columns) for _k, ds in items}
  if len(shapes) != 1:
    raise DicomLoadError("スライスごとに画像サイズが異なるため読み込めません。")

  slices = []
  for _k, ds in items:
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    inter = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = ds.pixel_array.astype(np.float32) * slope + inter
    slices.append(np.clip(np.rint(arr), -32768, 32767).astype(np.int16))
  data = np.stack(slices, axis=0)
  if progress:
    progress(total, total)

  first = items[0][1]
  ps = getattr(first, "PixelSpacing", [1.0, 1.0])
  sy, sx = float(ps[0]), float(ps[1])  # PixelSpacing = [行間隔, 列間隔]

  keys = np.array([k for k, _ds in items], dtype=float)
  if len(keys) > 1 and np.median(np.abs(np.diff(keys))) > 1e-6:
    sz = float(np.median(np.abs(np.diff(keys))))
  else:
    sz = float(getattr(first, "SliceThickness", 1.0) or 1.0)

  wc, ww = _default_window(first, data)
  return Volume(data, (sx, sy, sz), series.description, wc, ww)


def _default_window(ds, data: np.ndarray) -> Tuple[float, float]:
  def first_value(v):
    if v is None:
      return None
    try:
      return float(v[0]) if hasattr(v, "__len__") and not isinstance(v, str) else float(v)
    except Exception:
      return None

  wc = first_value(getattr(ds, "WindowCenter", None))
  ww = first_value(getattr(ds, "WindowWidth", None))
  if wc is None or ww is None or ww <= 0:
    wc, ww = 40.0, 400.0
  return wc, ww
