"""Provenance: everything needed to trace a video back to code, config and environment."""
from __future__ import annotations

import hashlib
import platform
import subprocess
from pathlib import Path
from typing import Any

from . import __version__


def git_commit(path: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain"], capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance_info(cfg) -> dict[str, Any]:
    info: dict[str, Any] = {
        "physscene_version": __version__,
        "physscene_commit": git_commit(Path(__file__).resolve().parent.parent),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "physics_backend": cfg.physics.backend,
        "physics_timestep": cfg.physics.timestep,
        "experiment": cfg.name,
        "seed": cfg.seed,
    }
    try:
        import mujoco

        info["mujoco_version"] = mujoco.__version__
    except ImportError:
        pass
    if cfg.source_path is not None:
        info["config_sha256"] = file_sha256(cfg.source_path)
        files = {"catalog": cfg.raw.get("catalog"), **{f"level:{i}": p for i, p in enumerate(cfg.raw.get("levels", []))}}
        base = cfg.source_path.parent
        info["input_sha256"] = {
            k: file_sha256(Path(v) if Path(v).is_absolute() else base / v) for k, v in files.items() if v
        }
    return info
