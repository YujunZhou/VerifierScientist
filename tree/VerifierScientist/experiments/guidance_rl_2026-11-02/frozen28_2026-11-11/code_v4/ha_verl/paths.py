"""Locate the HarnessAudit checkout and make it importable.

This integration treats HarnessAudit as a read-only library: nothing is
installed into it and no file inside it is modified. The repo is put on
``sys.path`` so ``import multi_agent`` resolves.

Override the location with ``HARNESSAUDIT_ROOT`` if the checkout moves.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

_DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "HarnessAudit-main"


@lru_cache(maxsize=1)
def harnessaudit_root() -> Path:
    root = Path(os.environ.get("HARNESSAUDIT_ROOT", str(_DEFAULT_ROOT))).resolve()
    if not (root / "multi_agent").is_dir():
        raise RuntimeError(
            f"HarnessAudit checkout not found at {root}. "
            "Set HARNESSAUDIT_ROOT to the repo containing multi_agent/."
        )
    return root


@lru_cache(maxsize=1)
def ensure_importable() -> Path:
    """Put the HarnessAudit repo on sys.path and return its root."""
    root = harnessaudit_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_DEFAULT_VERL_ROOT = Path(__file__).resolve().parents[2] / "workAmazon" / "verl_sglang"


@lru_cache(maxsize=1)
def ensure_verl_importable() -> Path:
    """Put the verl checkout on sys.path and return its root.

    The env's editable install of verl records a path that no longer exists, so
    ``import verl`` only works when the process happens to run from the checkout.
    Resolving it explicitly makes tools and tests independent of cwd. Override
    with ``VERL_ROOT``.
    """
    root = Path(os.environ.get("VERL_ROOT", str(_DEFAULT_VERL_ROOT))).resolve()
    if not (root / "verl").is_dir():
        raise RuntimeError(
            f"verl checkout not found at {root}. Set VERL_ROOT to the directory "
            "containing verl/."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def package_root() -> Path:
    """The ``multi_agent`` package directory."""
    return harnessaudit_root() / "multi_agent"


def fixtures_root() -> Path:
    return Path(os.environ.get("HA_FIXTURES_ROOT", str(harnessaudit_root() / "fixtures"))).resolve()


def workspaces_root() -> Path:
    return Path(os.environ.get("MASP_WORKSPACES_ROOT", str(package_root() / "workspaces")))
