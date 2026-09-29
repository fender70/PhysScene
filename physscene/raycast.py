"""Vectorised ray casting against simple primitives (numpy only).

Used for occlusion and visibility checks during planning, and by the
lightweight preview renderer that lets you dry-run the whole pipeline without
Unreal Engine.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .catalog import Shape
from .geometry import quat_to_matrix
from .spec import CameraSpec

EPS = 1e-6


@dataclass
class Primitive:
    name: str
    shape: Shape | None  # None = infinite horizontal plane at position z
    position: np.ndarray
    quat: np.ndarray
    color: tuple[float, float, float] = (0.7, 0.7, 0.7)
    stencil: int = 0


def _sphere(o, d, r):
    b = np.einsum("ij,ij->i", o, d)
    c = np.einsum("ij,ij->i", o, o) - r * r
    a = np.einsum("ij,ij->i", d, d)
    disc = b * b - a * c
    t = np.full(len(o), np.inf)
    ok = disc >= 0
    sq = np.sqrt(np.where(ok, disc, 0))
    t0 = (-b - sq) / a
    t1 = (-b + sq) / a
    tt = np.where(t0 > EPS, t0, np.where(t1 > EPS, t1, np.inf))
    t[ok] = tt[ok]
    return t


def _box(o, d, he):
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / np.where(np.abs(d) < 1e-12, 1e-12, d)
        t1 = (-he - o) * inv
        t2 = (he - o) * inv
    tmin = np.max(np.minimum(t1, t2), axis=1)
    tmax = np.min(np.maximum(t1, t2), axis=1)
    hit = tmax >= np.maximum(tmin, EPS)
    t = np.where(tmin > EPS, tmin, tmax)
    return np.where(hit, t, np.inf)


def _cylinder(o, d, r, h, capsule=False):
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = 2 * (o[:, 0] * d[:, 0] + o[:, 1] * d[:, 1])
    c = o[:, 0] ** 2 + o[:, 1] ** 2 - r * r
    disc = b * b - 4 * a * c
    t = np.full(len(o), np.inf)
    ok = (disc >= 0) & (a > 1e-12)
    sq = np.sqrt(np.where(ok, disc, 0))
    safe_a = np.where(a > 1e-12, a, 1)
    for root in ((-b - sq) / (2 * safe_a), (-b + sq) / (2 * safe_a)):
        z = o[:, 2] + root * d[:, 2]
        good = ok & (root > EPS) & (np.abs(z) <= h)
        t = np.where(good & (root < t), root, t)
    if capsule:
        for zc in (-h, h):
            oc = o - np.array([0, 0, zc])
            t = np.minimum(t, _sphere(oc, d, r))
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            for zc in (-h, h):
                tc = (zc - o[:, 2]) / np.where(np.abs(d[:, 2]) < 1e-12, 1e-12, d[:, 2])
                x = o[:, 0] + tc * d[:, 0]
                y = o[:, 1] + tc * d[:, 1]
                good = (tc > EPS) & (x * x + y * y <= r * r)
                t = np.where(good & (tc < t), tc, t)
    return t


def intersect(origins: np.ndarray, dirs: np.ndarray, prim: Primitive) -> np.ndarray:
    """Distance along each ray to ``prim`` (inf on miss). ``dirs`` need not be unit."""
    if prim.shape is None:  # horizontal plane
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (prim.position[2] - origins[:, 2]) / dirs[:, 2]
        return np.where((t > EPS) & np.isfinite(t), t, np.inf)
    r = quat_to_matrix(prim.quat)
    o = (origins - prim.position) @ r  # == R^T (o - c) row-wise
    d = dirs @ r
    s = prim.shape
    if s.type == "sphere":
        return _sphere(o, d, s.radius)
    if s.type == "box":
        return _box(o, d, np.asarray(s.half_extents))
    if s.type == "cylinder":
        return _cylinder(o, d, s.radius, s.half_length)
    if s.type == "capsule":
        return _cylinder(o, d, s.radius, s.half_length, capsule=True)
    raise ValueError(s.type)


def cast(origins, dirs, prims: list[Primitive]) -> tuple[np.ndarray, np.ndarray]:
    """Nearest hit over all primitives. Returns (t, prim_index) with index -1 on miss."""
    best_t = np.full(len(origins), np.inf)
    best_i = np.full(len(origins), -1, dtype=int)
    for i, p in enumerate(prims):
        t = intersect(origins, dirs, p)
        closer = t < best_t
        best_t[closer] = t[closer]
        best_i[closer] = i
    return best_t, best_i


def camera_rays(cam: CameraSpec, width: int | None = None, height: int | None = None):
    """World-space ray origins and (unit) directions, one per pixel, row-major."""
    w = width or cam.resolution[0]
    h = height or cam.resolution[1]
    fx = w / (2.0 * np.tan(np.radians(cam.hfov_deg) / 2.0))
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    local = np.stack([np.ones_like(u), -(u - w / 2) / fx, -(v - h / 2) / fx], axis=-1).reshape(-1, 3)
    r = quat_to_matrix(cam.quat)
    dirs = local @ r.T
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    origins = np.broadcast_to(np.asarray(cam.position, float), dirs.shape).copy()
    return origins, dirs


def project(cam: CameraSpec, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project world points to pixels. Returns (uv [N,2], depth [N]); depth<=0 = behind."""
    r = quat_to_matrix(cam.quat)
    pc = (np.asarray(points, float) - np.asarray(cam.position, float)) @ r  # (fwd, left, up)
    w, h = cam.resolution
    fx = w / (2.0 * np.tan(np.radians(cam.hfov_deg) / 2.0))
    depth = pc[:, 0]
    safe = np.where(np.abs(depth) < 1e-9, 1e-9, depth)
    u = w / 2 - fx * pc[:, 1] / safe
    v = h / 2 - fx * pc[:, 2] / safe
    return np.stack([u, v], axis=1), depth


def sample_surface_points(shape: Shape, position, quat, n: int = 64, rng=None) -> np.ndarray:
    """Points on the surface of a primitive (world frame), for visibility tests."""
    rng = rng or np.random.default_rng(0)
    he = shape.local_half_extents()
    if shape.type == "sphere":
        p = rng.normal(size=(n, 3))
        p = p / np.linalg.norm(p, axis=1, keepdims=True) * shape.radius
    elif shape.type == "box":
        p = rng.uniform(-1, 1, size=(n, 3)) * he
        ax = rng.integers(0, 3, size=n)
        p[np.arange(n), ax] = np.sign(rng.uniform(-1, 1, size=n)) * he[ax]
    else:
        th = rng.uniform(0, 2 * np.pi, size=n)
        z = rng.uniform(-shape.half_length, shape.half_length, size=n)
        p = np.stack([np.cos(th) * shape.radius, np.sin(th) * shape.radius, z], axis=1)
    p = p * 0.97  # pull slightly inside to avoid self-hit precision issues at the surface
    return p @ quat_to_matrix(quat).T + np.asarray(position, float)
