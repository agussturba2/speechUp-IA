"""Disk cache convenience wrappers.

Provides a single function ``get_cache`` that returns a ``diskcache.Cache``
instance.  All modules should obtain their cache through this utility to avoid
hard-coding paths throughout the codebase.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

from diskcache import Cache

_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "cache"


def get_cache(name: str = "results", base_dir: Union[str, os.PathLike] | None = None) -> Cache:
    """Return a ``diskcache.Cache`` located under *base_dir* / *name*.

    If *base_dir* is None, a project-level ``cache/`` folder is created
    alongside the source tree.
    """

    base_path = Path(base_dir) if base_dir is not None else _DEFAULT_DIR
    base_path.mkdir(parents=True, exist_ok=True)
    return Cache(base_path / name)
