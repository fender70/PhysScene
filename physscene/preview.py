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
from .raycast import Primitive, camera_rays, cast, intersect
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


def _same_prim(a: Primitive, b: Primitive) -> bool:
    return (
        a.name == b.name
        and a.shape == b.shape
        and a.color == b.color
        and a.stencil == b.stencil
        and np.array_equal(a.position, b.position)
        and np.array_equal(a.quat, b.quat)
    )


def render_frames(frames: list[list[Primitive]], cam: CameraSpec, width: int, height: int):
    """Render a clip from a fixed camera, ray casting the geometry that never
    moves only once.

    Gives the same pixels as calling :func:`render_frame` on every frame: a
    pixel's nearest hit is the smallest distance, ties going to the primitive
    listed first, exactly as in :func:`~physscene.raycast.cast`. Static hits,
    their shading and the shadows that static geometry casts on them are
    computed once. Each frame then casts only the moving bodies, plus their
    shadows on static surfaces."""
    first = frames[0]
    static_names = {
        p.name for p in first if all(any(_same_prim(p, q) for q in f) for f in frames[1:])
    }
    statics = [p for p in first if p.name in static_names]
    origins, dirs = camera_rays(cam, width, height)
    n = len(origins)
    fwd = quat_to_matrix(cam.quat)[:, 0]
    up = np.clip(dirs[:, 2] * 0.5 + 0.5, 0, 1)[:, None]
    sky = SKY_BOTTOM * (1 - up) + SKY_TOP * up

    # static layer: per primitive distances along the camera rays
    t_static = {p.name: intersect(origins, dirs, p) for p in statics}
    s_t = np.full(n, np.inf)
    s_name = np.full(n, -1, dtype=int)
    for i, p in enumerate(statics):
        closer = t_static[p.name] < s_t
        s_t[closer] = t_static[p.name][closer]
        s_name[closer] = i
    s_hit = s_name >= 0
    pts = origins[s_hit] + dirs[s_hit] * s_t[s_hit, None]
    nrm = np.zeros((s_hit.sum(), 3))
    col = np.zeros((s_hit.sum(), 3))
    for i, p in enumerate(statics):
        sel = s_name[s_hit] == i
        if sel.any():
            col[sel] = p.color
            nrm[sel] = _normals(pts[sel], p)
    flip = np.einsum("ij,ij->i", nrm, dirs[s_hit]) > 0
    nrm[flip] *= -1
    s_lambert = np.clip(nrm @ LIGHT_DIR, 0, 1)
    s_so = pts + nrm * 1e-3
    light_dirs = np.tile(LIGHT_DIR, (len(s_so), 1))
    s_shadow = np.zeros(len(s_so), bool)
    for p in statics:
        s_shadow |= np.isfinite(intersect(s_so, light_dirs, p))
    s_col = col
    s_depth = (s_t[s_hit] * (dirs[s_hit] @ fwd)).astype(np.float32)
    s_index = np.flatnonzero(s_hit)  # pixel -> row in the static arrays
    row_of = np.full(n, -1, dtype=int)
    row_of[s_index] = np.arange(len(s_index))

    for prims in frames:
        order = {p.name: k for k, p in enumerate(prims)}
        dyn = [p for p in prims if p.name not in static_names]
        # nearest moving body, ties to the one listed first
        d_t = np.full(n, np.inf)
        d_i = np.full(n, -1, dtype=int)
        for p in dyn:
            t = intersect(origins, dirs, p)
            closer = t < d_t
            d_t[closer] = t[closer]
            d_i[closer] = order[p.name]
        s_order = np.where(s_hit, np.array([order[p.name] for p in statics] + [n])[s_name], n)
        use_dyn = (d_t < s_t) | ((d_t == s_t) & (d_i < s_order) & np.isfinite(d_t))
        rgb = sky.copy()
        stencil = np.zeros(n, dtype=np.uint8)
        depth = np.full(n, np.inf, dtype=np.float32)

        # static pixels: cached shading, plus shadows cast by moving bodies
        sp = s_hit & ~use_dyn
        rows = row_of[sp]
        shadow = s_shadow[rows].copy()
        if dyn:
            so = s_so[rows]
            ld = light_dirs[: len(rows)]
            for p in dyn:
                shadow |= np.isfinite(intersect(so, ld, p))
        light = np.where(shadow, 0.35, 1.0)
        shade = 0.28 + 0.72 * s_lambert[rows] * light
        rgb[sp] = s_col[rows] * shade[:, None]
        depth[sp] = s_depth[rows]
        stencil[sp] = np.array([p.stencil for p in statics], dtype=np.uint8)[s_name[sp]]

        # pixels where a moving body is nearest: shaded as in render_frame
        dp = use_dyn
        if dp.any():
            pts_d = origins[dp] + dirs[dp] * d_t[dp, None]
            depth[dp] = (d_t[dp] * (dirs[dp] @ fwd)).astype(np.float32)
            idx = d_i[dp]
            col_d = np.zeros((dp.sum(), 3))
            nrm_d = np.zeros((dp.sum(), 3))
            for p in dyn:
                sel = idx == order[p.name]
                if sel.any():
                    col_d[sel] = p.color
                    nrm_d[sel] = _normals(pts_d[sel], p)
            stencil[dp] = np.array([p.stencil for p in prims], dtype=np.uint8)[idx]
            flip = np.einsum("ij,ij->i", nrm_d, dirs[dp]) > 0
            nrm_d[flip] *= -1
            lam = np.clip(nrm_d @ LIGHT_DIR, 0, 1)
            st, _ = cast(pts_d + nrm_d * 1e-3, np.tile(LIGHT_DIR, (len(pts_d), 1)), prims)
            rgb[dp] = col_d * (0.28 + 0.72 * lam * np.where(np.isfinite(st), 0.35, 1.0))[:, None]
        rgb = np.clip(rgb, 0, 1) ** (1 / 1.6)
        yield (
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
    frames = [scene_primitives(layout, traj, f, level, catalog, colors) for f in range(traj.num_frames)]
    for f, (rgb, stencil, depth) in enumerate(render_frames(frames, cam, w, h)):
        Image.fromarray(rgb).save(out_dir / "rgb" / f"{f:04d}.png")
        if "mask" in passes:
            Image.fromarray(stencil_to_bits_rgb(stencil)).save(out_dir / "labels" / f"{f:04d}.png")
        if "depth" in passes:
            np.savez_compressed(out_dir / "depth" / f"{f:04d}.npz", depth=depth.astype(np.float16))
    (out_dir / "DONE").write_text(job_fingerprint(job) + " preview\n")
