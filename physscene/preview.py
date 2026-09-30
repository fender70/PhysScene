"""Lightweight preview renderer (numpy ray casting, no GPU and no Unreal).

It renders every job with flat-shaded primitives and writes the **same raw
layout** the Unreal driver writes (``rgb/``, ``labels/``, ``depth/``). That
lets you dry-run and debug the full pipeline (layouts, cameras, events,
masks, export, validation) before spending GPU hours in Unreal. It also
serves as a cheap "abstract" rendering style for ablations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .catalog import Catalog, Level
from .geometry import quat_to_matrix
from .raycast import Primitive, camera_rays, cast
from .scene import scene_primitives
from .spec import CameraSpec, Layout, Trajectory, job_fingerprint

LIGHT_DIR = np.array([0.35, -0.45, 0.82])
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)
SKY_TOP = np.array([0.55, 0.7, 0.9])
SKY_BOTTOM = np.array([0.85, 0.88, 0.92])


def _normals(points: np.ndarray, prim: Primitive) -> np.ndarray:
    if prim.shape is None:
        return np.tile([0.0, 0.0, 1.0], (len(points), 1))
    r = quat_to_matrix(prim.quat)
    lp = (points - prim.position) @ r
    s = prim.shape
    if s.type == "sphere":
        n = lp
    elif s.type == "box":
        he = np.asarray(s.half_extents)
        k = np.argmax(np.abs(lp) / he, axis=1)
        n = np.zeros_like(lp)
        n[np.arange(len(lp)), k] = np.sign(lp[np.arange(len(lp)), k])
    else:
        h = s.half_length
        n = lp.copy()
        if s.type == "capsule":
            n[:, 2] = lp[:, 2] - np.clip(lp[:, 2], -h, h)
        else:
            cap = np.abs(lp[:, 2]) >= h - 1e-4
            n[:, 2] = 0.0
            n[cap] = np.stack([np.zeros(cap.sum()), np.zeros(cap.sum()), np.sign(lp[cap, 2])], axis=1)
    n = n @ r.T
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)


def render_frame(prims: list[Primitive], cam: CameraSpec, width: int, height: int, shadows: bool = True):
    origins, dirs = camera_rays(cam, width, height)
    t, idx = cast(origins, dirs, prims)
    hit = idx >= 0
    # sky gradient
    up = np.clip(dirs[:, 2] * 0.5 + 0.5, 0, 1)[:, None]
    rgb = SKY_BOTTOM * (1 - up) + SKY_TOP * up
    stencil = np.zeros(len(origins), dtype=np.uint8)
    depth = np.full(len(origins), np.inf, dtype=np.float32)
    fwd = quat_to_matrix(cam.quat)[:, 0]
    if hit.any():
        pts = origins[hit] + dirs[hit] * t[hit, None]
        depth[hit] = (t[hit] * (dirs[hit] @ fwd)).astype(np.float32)
        col = np.zeros((hit.sum(), 3))
        nrm = np.zeros((hit.sum(), 3))
        hidx = idx[hit]
        for i, p in enumerate(prims):
            sel = hidx == i
            if not sel.any():
                continue
            col[sel] = p.color
            nrm[sel] = _normals(pts[sel], p)
        stencil[hit] = np.array([p.stencil for p in prims], dtype=np.uint8)[hidx]
        # face the camera
        flip = np.einsum("ij,ij->i", nrm, dirs[hit]) > 0
        nrm[flip] *= -1
        lambert = np.clip(nrm @ LIGHT_DIR, 0, 1)
        light = np.ones(len(pts))
        if shadows:
            so = pts + nrm * 1e-3
            st, _ = cast(so, np.tile(LIGHT_DIR, (len(so), 1)), prims)
            light = np.where(np.isfinite(st), 0.35, 1.0)
        shade = 0.28 + 0.72 * lambert * light
        rgb[hit] = col * shade[:, None]
    rgb = np.clip(rgb, 0, 1) ** (1 / 1.6)
    return (
        (rgb.reshape(height, width, 3) * 255).astype(np.uint8),
        stencil.reshape(height, width),
        depth.reshape(height, width),
    )


def stencil_to_bits_rgb(stencil: np.ndarray) -> np.ndarray:
    """Encode stencil ids 0..7 as RGB bits, like the Unreal stencil material does."""
    out = np.zeros((*stencil.shape, 3), dtype=np.uint8)
    for b in range(3):
        out[..., b] = ((stencil >> b) & 1) * 255
    return out


def render_job(job: dict[str, Any], plan_dir: Path, out_dir: Path, catalog: Catalog, level: Level, scale: float = 0.25) -> None:
    traj = Trajectory.from_dict(json.loads((plan_dir / job["trajectory"]).read_text()))
    layout_path = (plan_dir / job["trajectory"]).parent / "layout.json"
    layout = Layout.from_dict(json.loads(layout_path.read_text()))
    cam = CameraSpec.from_dict({k: job["camera"][k] for k in ("position", "quat", "hfov_deg", "resolution", "target")})
    w = max(16, int(round(cam.resolution[0] * scale)))
    h = max(16, int(round(cam.resolution[1] * scale)))
    colors = {b["name"]: tuple(b["preview_color"]) for b in job["bodies"]}
    passes = job["render"]["passes"]
    for sub in ("rgb", "labels", "depth"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    for f in range(traj.num_frames):
        prims = scene_primitives(layout, traj, f, level, catalog, colors)
        rgb, stencil, depth = render_frame(prims, cam, w, h)
        Image.fromarray(rgb).save(out_dir / "rgb" / f"{f:04d}.png")
        if "mask" in passes:
            Image.fromarray(stencil_to_bits_rgb(stencil)).save(out_dir / "labels" / f"{f:04d}.png")
        if "depth" in passes:
            np.savez_compressed(out_dir / "depth" / f"{f:04d}.npz", depth=depth.astype(np.float16))
    (out_dir / "DONE").write_text(job_fingerprint(job) + " preview\n")
