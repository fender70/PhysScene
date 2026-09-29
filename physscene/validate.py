"""Dataset validator: checks every sample is complete and usable by the CRONOS
evaluation code (and by PhysScene's own tools)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

REQUIRED = ["metadata.json", "movies/complete.mp4", "mask/frame_0000.jpg"]
RECOMMENDED = ["camera.json", "states.json", "masks.npz"]
META_KEYS = ["event", "scene", "object"]


@dataclass
class Report:
    samples: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        lines = [f"{self.samples} samples, {len(self.errors)} errors, {len(self.warnings)} warnings"]
        lines += [f"ERROR  {e}" for e in self.errors[:50]]
        lines += [f"WARN   {w}" for w in self.warnings[:50]]
        return "\n".join(lines)


def validate_sample(d: Path, rep: Report, root: Path) -> None:
    rel = d.relative_to(root)
    for f in REQUIRED:
        if not (d / f).exists():
            rep.errors.append(f"{rel}: missing {f}")
    for f in RECOMMENDED:
        if not (d / f).exists():
            rep.warnings.append(f"{rel}: missing {f}")
    if not (d / "metadata.json").exists():
        return
    meta = json.loads((d / "metadata.json").read_text())
    for k in META_KEYS:
        if k not in meta:
            rep.errors.append(f"{rel}: metadata.json lacks {k!r}")
    m0 = d / "mask" / "frame_0000.jpg"
    if m0.exists():
        img = np.asarray(Image.open(m0))
        obj = img if img.ndim == 2 else img[..., 0]
        if (obj > 127).sum() == 0:
            rep.errors.append(f"{rel}: object mask empty at frame 0")
    if (d / "masks.npz").exists():
        z = np.load(d / "masks.npz")
        t = z["masks"].shape[0]
        if t != meta.get("num_frames", t):
            rep.errors.append(f"{rel}: mask frames {t} != num_frames {meta.get('num_frames')}")
    parts = rel.parts
    if "alternatives" not in parts and len(parts) == 5:
        expected = [meta.get("event"), meta.get("scene"), meta.get("object")]
        if list(parts[:3]) != expected:
            rep.errors.append(f"{rel}: path does not match metadata {expected}")


def validate_dataset(root: str | Path) -> Report:
    root = Path(root)
    rep = Report()
    for mp in sorted(root.rglob("metadata.json")):
        rep.samples += 1
        validate_sample(mp.parent, rep, root)
    if rep.samples == 0:
        rep.errors.append(f"no samples (metadata.json) found under {root}")
    return rep
