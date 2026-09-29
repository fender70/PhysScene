"""Event-aware static camera sampling (the viewpoint intervention axis).

Each view is a static look-at camera placed on a sphere around the event. The
azimuth is stratified across views so ``views: N`` gives N distinct viewpoints.
The distance is solved so the whole trajectory stays in frame. For occlusion
events the azimuth is constrained so the occluder sits between the camera and
the object's path.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .catalog import Catalog, Level
from .geometry import look_at_matrix, matrix_to_quat
from .raycast import project
from .spec import CameraSpec, Layout, Trajectory

DEFAULT_CAMERA: dict[str, Any] = {
    "side_view_range_deg": 70.0,  # azimuth spread around the perpendicular to the motion
    "occlusion_range_deg": 22.0,  # azimuth spread around the occluder line of sight
    "fit_margin": 0.85,  # fraction of the image the trajectory may occupy
    "distance_jitter": [1.0, 1.25],
    # Low cameras for occlusion so the occluder actually hides the object.
    "occlusion_elevation_deg": [3.0, 14.0],
}


def _key_points(layout: Layout, traj: Trajectory, catalog: Catalog) -> np.ndarray:
    pts = [traj.pos("object")]
    for b in layout.bodies:
        if b.name != "object":
            pts.append(traj.pos(b.name)[[0, -1]])
    pts = np.concatenate(pts, axis=0)
    r = catalog.get(layout.body("object").asset).shape.bounding_radius()
    offs = np.array([[dx, dy, dz] for dx in (-r, r) for dy in (-r, r) for dz in (-r, r)])
    return (pts[:, None, :] + offs[None]).reshape(-1, 3)


def _make_cam(target, az, el, dist, hfov, resolution) -> CameraSpec:
    offset = dist * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    pos = np.asarray(target) + offset
    q = matrix_to_quat(look_at_matrix(pos, target))
    return CameraSpec(position=pos.tolist(), quat=q.tolist(), hfov_deg=float(hfov), resolution=list(resolution), target=list(map(float, target)))


def _fits(cam: CameraSpec, pts: np.ndarray, margin: float) -> bool:
    uv, depth = project(cam, pts)
    if np.any(depth <= 0.05):
        return False
    w, h = cam.resolution
    mx, my = w * (1 - margin) / 2, h * (1 - margin) / 2
    return bool(uv[:, 0].min() >= mx and uv[:, 0].max() <= w - mx and uv[:, 1].min() >= my and uv[:, 1].max() <= h - my)


def _camera_valid(cam: CameraSpec, level: Level) -> bool:
    p = np.asarray(cam.position)
    cc = level.camera
    if p[2] < level.floor_z + cc.min_height:
        return False
    if cc.bounds_min is not None and np.any(p < np.asarray(cc.bounds_min)):
        return False
    if cc.bounds_max is not None and np.any(p > np.asarray(cc.bounds_max)):
        return False
    for s in level.surfaces.values():
        if s.contains_xy(p) and s.top_z - s.thickness - 0.05 <= p[2] <= s.top_z + 0.05:
            return False
    return True


def sample_camera(
    rng: np.random.Generator,
    layout: Layout,
    traj: Trajectory,
    level: Level,
    catalog: Catalog,
    view_index: int,
    num_views: int,
    resolution,
    overrides: dict[str, Any] | None = None,
) -> CameraSpec | None:
    """Propose one camera for ``view_index`` of ``num_views``. Returns ``None``
    if no distance within the level's limits keeps the event in frame."""
    p = dict(DEFAULT_CAMERA)
    p.update(overrides or {})
    cc = level.camera
    pts = _key_points(layout, traj, catalog)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    target = (lo + hi) / 2

    d = np.asarray(layout.meta.get("direction", [1.0, 0.0, 0.0]), float)
    motion_az = math.atan2(d[1], d[0])
    strata = (view_index + rng.uniform(0.15, 0.85)) / max(1, num_views)  # in (0, 1)
    if layout.event == "occlusion":
        side = np.asarray(layout.meta["occluder_side"], float)
        base = math.atan2(side[1], side[0])  # camera sits on the occluder side of the path
        spread = math.radians(p["occlusion_range_deg"])
        az = base + (2 * strata - 1) * spread
        # Look at the occluder so it is centred between camera and path.
        occ = traj.pos("occluder")[0]
        target = np.array([0.5 * (target[0] + occ[0]), 0.5 * (target[1] + occ[1]), target[2]])
    else:
        side_sign = 1.0 if rng.uniform() < 0.5 else -1.0
        spread = math.radians(p["side_view_range_deg"])
        az = motion_az + side_sign * math.pi / 2 + (2 * strata - 1) * spread

    el_range = p["occlusion_elevation_deg"] if layout.event == "occlusion" else cc.elevation_deg
    el = math.radians(rng.uniform(*el_range))
    hfov = rng.uniform(*cc.hfov_deg)

    dist = None
    for dd in np.linspace(cc.distance[0], cc.distance[1], 40):
        cam = _make_cam(target, az, el, dd, hfov, resolution)
        if _fits(cam, pts, p["fit_margin"]):
            dist = dd
            break
    if dist is None:
        return None
    dist = min(cc.distance[1], dist * rng.uniform(*p["distance_jitter"]))
    cam = _make_cam(target, az, el, dist, hfov, resolution)
    if not _camera_valid(cam, level):
        return None
    return cam
