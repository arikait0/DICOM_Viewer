"""肺マスク(TIFF)の読み込みモジュール(GUI に依存しない)。

対応する形式:
  1. マルチページ TIFF 1 ファイル(全スライスが 1 ファイルに入っているもの)
  2. TIFF が 1 スライス 1 ファイルで入ったフォルダ(ファイル名の数字順に並べる)
値は 0 = 背景、0 以外 = 肺領域として扱う(0/1 でも 0/255 でも可)。
戻り値は bool 配列 (z, y, x)。
"""
from __future__ import annotations

import os
import re
from typing import List

import numpy as np
import tifffile

TIF_EXT = (".tif", ".tiff")


class MaskLoadError(Exception):
  """読み込み失敗時に GUI へ伝えるための例外。"""


def _natural_key(s: str):
  return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _read_tiff(path: str) -> np.ndarray:
  """TIFF 1 ファイルを (n, y, x) で返す(単一画像なら n=1)。"""
  try:
    with tifffile.TiffFile(path) as tf:
      series = tf.series[0]
      arr, axes = series.asarray(), series.axes
  except Exception as e:  # 圧縮方式が未対応などもここに来る
    raise MaskLoadError(
        f"TIFF を読み込めませんでした: {os.path.basename(path)}\n{e}")
  if "S" in axes:  # RGB などのサンプル軸は最大値でまとめる
    arr = arr.max(axis=axes.index("S"))
  if arr.ndim == 2:
    arr = arr[None]
  if arr.ndim != 3:
    raise MaskLoadError(
        f"対応していない次元数です: {os.path.basename(path)} (shape={arr.shape})")
  return arr


def load_mask(path: str) -> np.ndarray:
  """ファイルまたはフォルダから肺マスク (z, y, x) の bool 配列を作る。"""
  if os.path.isdir(path):
    files: List[str] = sorted(
        (f for f in os.listdir(path) if f.lower().endswith(TIF_EXT)),
        key=_natural_key)
    if not files:
      raise MaskLoadError("フォルダ内に .tif / .tiff ファイルがありません。")
    parts = [_read_tiff(os.path.join(path, f)) for f in files]
    shapes = {p.shape[1:] for p in parts}
    if len(shapes) != 1:
      raise MaskLoadError("ファイルごとに画像サイズが異なるため読み込めません。")
    arr = np.concatenate(parts, axis=0)
  else:
    arr = _read_tiff(path)
  return arr > 0
