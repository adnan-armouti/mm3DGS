"""v6 data preprocessing pipeline (port of mmir/preprocessing).

This package mirrors the layout and intent of ``mmir/preprocessing`` —
config generation, radar-LiDAR alignment, scene/mesh build — with
minimal edits to support:

  (a) a ``data_v2/`` output tree (explicit ``--out-root`` CLI),
  (b) per-scene extended radar-frame windows (larger than the
      existing 9-frame default), including a chunked fused-LiDAR path
      so that ~70-frame cascade windows fit in RAM.

The alignment/ subdirectory was already ported here earlier (pass-2 +
pass-3 edits). This package init adds the preprocessing dir itself to
``sys.path`` so that the submodule-style absolute imports inherited
from ``mmir/preprocessing`` (``from ColoRadar_tools.dataset_loaders
import ...``, ``from io_paths import ...``, etc.) resolve without
modification.

DO NOT EDIT ``mmir/preprocessing/`` — all local changes live here.
"""
import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
