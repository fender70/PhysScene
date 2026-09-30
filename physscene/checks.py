"""Automatic verification that a simulated sample shows the intended event.

In CRONOS each scene was adjusted by hand until the event "looked right".
Here every proposal is checked programmatically and rejected if it fails, so
generation can run unattended.

* :func:`check_physics`: camera-independent checks on the trajectory, e.g. did
  the object leave the table and hit the floor, did it touch the collider.
* :func:`check_view`: camera-dependent checks, e.g. the object is in frame and
  unoccluded at frame 0, is fully hidden by the occluder mid-clip and
  reappears.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .catalog import Catalog, Level
from .raycast import cast, project, sample_surface_points
from .scene import scene_primitives
from .spec import CameraSpec, Layout, Trajectory

DEFAULT_CHECKS: dict[str, Any] = {
    "event_window": [0.12, 0.88],  # the key moment must fall inside this fraction of the clip
    "min_in_frame_frac": 0.95,
    "frame_margin_px": 4,
    "min_object_px": 14,
    "first_frame_visibility": 0.9,
    "occlusion_max_visibility": 0.1,
    "reappear_visibility": 0.6,
    "general_min_visibility": 0.5,
    "visibility_samples": 48,
    # A collision must visibly move a dynamic collider (m); 0 disables the check.
    "min_collider_displacement": 0.01,
    "min_velocity_change": 0.25,  # m/s change of the object's velocity at impact
}


@dataclass
class CheckResult:
    ok: bool
    reason: str = ""
    info: dict[str, Any] = field(default_factory=dict)
    hint: str = ""  # "faster" / "slower": how to adjust the initial push on failure

    def __bool__(self) -> bool:
        return self.ok


def _params(overrides: dict[str, Any] | None) -> dict[str, Any]:
    p = dict(DEFAULT_CHECKS)
    p.update(overrides or {})
    return p


def _on_surface(level: Level, layout: Layout, p: np.ndarray, tol: float = 0.0) -> bool:
    s = level.surfaces[layout.surface]
    return s.contains_xy(p, margin=-tol)


def check_physics(
    layout: Layout, traj: Trajectory, level: Level, catalog: Catalog, overrides: dict[str, Any] | None = None
) -> CheckResult:
    p = _params(overrides)
    s = level.surfaces[layout.surface]
    obj = traj.pos("object")
    T = len(obj)
    lo, hi = int(p["event_window"][0] * T), int(p["event_window"][1] * T)
    asset = catalog.get(layout.body("object").asset)
    r = asset.shape.bounding_radius()

    if not np.all(np.isfinite(obj)):
        return CheckResult(False, "simulation diverged")

    if layout.event == "fall":
        left = [
            t for t in range(T) if not s.contains_xy(obj[t]) and obj[t, 2] < s.top_z - 0.25 * r
        ]
        if not left:
            return CheckResult(False, "object never fell off the surface", hint="faster")
        t_fall = left[0]
        if not (lo <= t_fall <= hi):
            return CheckResult(False, f"fall at frame {t_fall} outside window [{lo},{hi}]", hint="faster" if t_fall > hi else "slower")
        landed = [t for t in range(t_fall, T) if obj[t, 2] < level.floor_z + 1.5 * r]
        if not landed:
            return CheckResult(False, "object did not reach the floor before the clip ended", hint="faster")
        return CheckResult(True, info={"event_frame": t_fall, "landing_frame": landed[0]})

    if layout.event == "collision":
        hits = [t for t, cs in enumerate(traj.contacts) if ["collider", "object"] in cs]
        if not hits:
            return CheckResult(False, "no contact between object and collider", hint="faster")
        t_hit = hits[0]
        if not (lo <= t_hit <= hi):
            return CheckResult(False, f"collision at frame {t_hit} outside window [{lo},{hi}]", hint="faster" if t_hit > hi else "slower")
        if not s.contains_xy(obj[t_hit]):
            return CheckResult(False, "object left the surface before colliding")
        if p["min_collider_displacement"] > 0:
            # Visible either as the collider moving, or as the object bouncing / deflecting.
            col = traj.pos("collider")
            moved = float(np.max(np.linalg.norm(col - col[0], axis=1)))
            a, b = max(0, t_hit - 3), min(T - 1, t_hit + 3)
            v_before = (obj[t_hit] - obj[a]) / max(1, t_hit - a)
            v_after = (obj[b] - obj[t_hit]) / max(1, b - t_hit)
            dv = float(np.linalg.norm(v_after - v_before)) * traj.fps
            if moved < p["min_collider_displacement"] and dv < p["min_velocity_change"]:
                return CheckResult(
                    False, f"collision too gentle (collider moved {moved * 100:.1f} cm, object dv {dv:.2f} m/s)", hint="faster"
                )
        return CheckResult(True, info={"event_frame": t_hit})

    if layout.event == "occlusion":
        if not all(s.contains_xy(obj[t], margin=0.0) for t in range(T)):
            return CheckResult(False, "object left the surface during occlusion clip", hint="slower")
        if np.any(obj[:, 2] < s.top_z):
            return CheckResult(False, "object sank / fell")
        d = np.asarray(layout.meta["direction"], float)
        occ = traj.pos("occluder")[0]
        occ_asset = catalog.get(layout.body("occluder").asset)
        rp = occ_asset.shape.bounding_radius()
        along_obj = (obj - occ) @ d
        if along_obj[-1] < r + 0.5 * rp:
            return CheckResult(False, "object did not pass the occluder", hint="faster")
        t_pass = int(np.argmin(np.abs(along_obj)))
        if not (lo <= t_pass <= hi):
            return CheckResult(False, f"occluder crossing at frame {t_pass} outside window", hint="faster" if t_pass > hi else "slower")
        return CheckResult(True, info={"event_frame": t_pass})

    # Custom events: accept, and let a user-supplied check (if any) decide.
    return CheckResult(True, info={})


def visibility(
    layout: Layout,
    traj: Trajectory,
    frame: int,
    cam: CameraSpec,
    level: Level,
    catalog: Catalog,
    body: str = "object",
    n: int = 48,
) -> float:
    """Fraction of sample points on ``body`` whose camera ray hits ``body`` first."""
    prims = scene_primitives(layout, traj, frame, level, catalog)
    idx = next(i for i, pr in enumerate(prims) if pr.name == body)
    pr = prims[idx]
    pts = sample_surface_points(pr.shape, pr.position, pr.quat, n=n)
    origin = np.asarray(cam.position, float)
    dirs = pts - origin
    _, hit = cast(np.broadcast_to(origin, dirs.shape).copy(), dirs, prims)
    return float(np.mean(hit == idx))


def projected_extent(cam: CameraSpec, catalog: Catalog, layout: Layout, traj: Trajectory, frame: int, body="object"):
    b = layout.body(body)
    a = catalog.get(b.asset)
    pts = sample_surface_points(a.shape, traj.positions[body][frame], traj.quats[body][frame], n=64)
    uv, depth = project(cam, pts)
    return uv, depth


def check_view(
    layout: Layout,
    traj: Trajectory,
    cam: CameraSpec,
    level: Level,
    catalog: Catalog,
    phys_info: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> CheckResult:
    p = _params(overrides)
    w, h = cam.resolution
    m = p["frame_margin_px"]
    T = traj.num_frames

    # Frame 0: object fully in frame, big enough, unoccluded.
    uv, depth = projected_extent(cam, catalog, layout, traj, 0)
    if np.any(depth <= 0):
        return CheckResult(False, "object behind camera at frame 0")
    if uv[:, 0].min() < m or uv[:, 1].min() < m or uv[:, 0].max() > w - m or uv[:, 1].max() > h - m:
        return CheckResult(False, "object not fully in frame at frame 0")
    size = max(np.ptp(uv[:, 0]), np.ptp(uv[:, 1]))
    if size < p["min_object_px"]:
        return CheckResult(False, f"object too small at frame 0 ({size:.1f}px)")
    n = p["visibility_samples"]
    v0 = visibility(layout, traj, 0, cam, level, catalog, n=n)
    if v0 < p["first_frame_visibility"]:
        return CheckResult(False, f"object occluded at frame 0 (visibility {v0:.2f})")

    # All frames: object centre stays in frame.
    centers, cdepth = project(cam, traj.pos("object"))
    inside = (cdepth > 0) & (centers[:, 0] >= 0) & (centers[:, 0] < w) & (centers[:, 1] >= 0) & (centers[:, 1] < h)
    if inside.mean() < p["min_in_frame_frac"]:
        return CheckResult(False, f"object in frame only {inside.mean():.0%} of the clip")

    # Other dynamic bodies at frame 0 should be at least partly visible.
    for b in layout.bodies:
        if b.name != "object" and b.stencil > 0:
            if visibility(layout, traj, 0, cam, level, catalog, body=b.name, n=n) < 0.3:
                return CheckResult(False, f"{b.name} hidden at frame 0")

    stride = max(1, T // 24)
    frames = list(range(0, T, stride))
    vis = {t: visibility(layout, traj, t, cam, level, catalog, n=n) for t in frames}
    info: dict[str, Any] = {"visibility": {str(k): round(v, 3) for k, v in vis.items()}}

    if layout.event == "occlusion":
        mid = [t for t in frames if 0.15 * T <= t <= 0.85 * T]
        vmin = min(vis[t] for t in mid) if mid else 1.0
        if vmin > p["occlusion_max_visibility"]:
            return CheckResult(False, f"object never hidden (min visibility {vmin:.2f})", info)
        t_hidden = min(mid, key=lambda t: vis[t])
        after = [vis[t] for t in frames if t > t_hidden]
        v_end = visibility(layout, traj, T - 1, cam, level, catalog, n=n)
        if max(after + [v_end]) < p["reappear_visibility"]:
            return CheckResult(False, "object does not reappear", info)
        info["hidden_frame"] = t_hidden
    else:
        ok_frac = np.mean([v >= p["general_min_visibility"] for v in vis.values()])
        if ok_frac < 0.85:
            return CheckResult(False, f"object mostly occluded ({ok_frac:.0%} of frames visible)", info)
    return CheckResult(True, info=info)
