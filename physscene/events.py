"""Event templates: procedural layouts for canonical physical events.

A template turns (level, surface, object, prop, rng, params) into a
:class:`~physscene.spec.Layout`, i.e. the initial conditions of every body. It
replaces manual per-level object placement and impulse tuning.

Templates only **propose** layouts. The planner simulates each proposal and
keeps it only if :mod:`physscene.checks` confirms the event actually happened.

To add your own event, register a function::

    @register_event("slide_and_stop")
    def slide_and_stop(ctx: EventContext) -> Layout: ...
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .catalog import Asset, Level, Surface
from .geometry import quat_from_axis_angle, quat_from_yaw, quat_mul, quat_to_matrix, rotate
from .spec import BodyInit, Layout

EVENT_REGISTRY: dict[str, Callable[["EventContext"], Layout]] = {}

# Defaults per event; override them under ``events.<name>`` in the experiment config.
DEFAULTS: dict[str, dict[str, Any]] = {
    "common": {
        "event_time_frac": [0.35, 0.6],  # when the key moment happens within the clip
        "max_heading_jitter_deg": 12.0,
        "max_speed": 3.0,
        "min_speed": 0.15,
        "edge_margin": 0.03,
        # Heading of collision/occlusion runs: along the surface's long axis +- jitter, or uniform.
        "heading_mode": "long_axis",
        "long_axis_jitter_deg": 35.0,
    },
    # Placement is *kinematic* by default: the template picks when the key
    # moment happens (event_time_frac) and how fast the object is moving at that
    # moment, then solves for the push and the distance from an estimate of the
    # object's deceleration. Setting distance_to_edge / distance / travel_frac
    # switches to the older fixed-distance placement.
    "fall": {"edge_speed": [0.25, 0.6]},  # m/s when rolling off the edge
    "collision": {"impact_speed": [0.5, 1.2], "lateral_offset_frac": [-0.3, 0.3], "props": []},
    "occlusion": {
        "pass_speed": [0.2, 0.45],  # m/s when passing behind the occluder
        "occluder_clearance": [0.04, 0.15],  # lateral gap between object path and occluder
        "occluder_position_frac": [0.4, 0.6],  # distance placement only
        "props": [],
    },
}


@dataclass
class EventContext:
    rng: np.random.Generator
    level: Level
    surface: Surface
    obj: Asset
    prop: Asset | None
    clip_duration: float
    params: dict[str, Any]

    def p(self, key: str):
        return self.params[key]

    def uniform(self, key: str) -> float:
        lo, hi = self.params[key]
        return float(self.rng.uniform(lo, hi))


def register_event(name: str):
    def deco(fn):
        EVENT_REGISTRY[name] = fn
        return fn

    return deco


def event_params(event: str, overrides: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(DEFAULTS["common"])
    out.update(DEFAULTS.get(event, {}))
    out.update(overrides or {})
    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _spin_axis(asset: Asset) -> np.ndarray | None:
    if asset.shape.type == "sphere":
        return None  # fully random
    if asset.shape.type in ("cylinder", "capsule"):
        return np.array([0.0, 0.0, 1.0])
    return None


def oriented_quat(asset: Asset, heading: float, rng: np.random.Generator, randomize_spin: bool = True) -> np.ndarray:
    """Orientation where ``asset.forward_axis`` (after its rest rotation) points
    along world heading ``heading`` (radians)."""
    q = np.asarray(asset.rest_rotation, float)
    if randomize_spin:
        if asset.shape.type == "sphere":
            a = rng.normal(size=3)
            q = quat_mul(q, quat_from_axis_angle(a, rng.uniform(0, 2 * math.pi)))
        elif (ax := _spin_axis(asset)) is not None:
            q = quat_mul(q, quat_from_axis_angle(ax, rng.uniform(0, 2 * math.pi)))
    f = rotate(np.asarray(asset.rest_rotation, float), asset.forward_axis)
    base = math.atan2(f[1], f[0])
    return quat_mul(quat_from_yaw(heading - base), q)


def resting_position(asset: Asset, quat: np.ndarray, surface_point: np.ndarray) -> np.ndarray:
    z = surface_point[2] + asset.shape.support_height(quat) + 1e-3
    return np.array([surface_point[0], surface_point[1], z])


def rolling_state(asset: Asset, quat: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """Angular velocity for rolling without slipping (zero for sliders)."""
    if asset.motion != "roll":
        return np.zeros(3)
    h = asset.shape.support_height(quat)
    return np.cross([0.0, 0.0, 1.0], velocity) / max(h, 1e-4)


def footprint_radius(asset: Asset) -> float:
    """Horizontal radius of the asset in its resting orientation (any yaw)."""
    he = asset.shape.local_half_extents()
    if asset.shape.type == "sphere":
        return float(he[0])
    ext = np.abs(quat_to_matrix(asset.rest_rotation)) @ he  # world-axis half extents
    if asset.shape.type == "box":
        return float(math.hypot(ext[0], ext[1]))
    return float(max(ext[0], ext[1]))


def _make_object(ctx: EventContext, start_local, heading_local: float, speed: float) -> BodyInit:
    s = ctx.surface
    heading = heading_local + math.radians(s.yaw_deg)
    q = oriented_quat(ctx.obj, heading, ctx.rng)
    pos = resting_position(ctx.obj, q, s.to_world(start_local))
    v = speed * np.array([math.cos(heading), math.sin(heading), 0.0])
    w = rolling_state(ctx.obj, q, v)
    return BodyInit(
        name="object",
        asset=ctx.obj.key,
        role="object",
        position=pos.tolist(),
        quat=q.tolist(),
        lin_vel=v.tolist(),
        ang_vel=w.tolist(),
        dynamic=True,
        stencil=1,
    )


def _make_prop(ctx: EventContext, local_xy, role: str, stencil: int) -> BodyInit:
    assert ctx.prop is not None
    s = ctx.surface
    q = oriented_quat(ctx.prop, ctx.rng.uniform(0, 2 * math.pi), ctx.rng, randomize_spin=False)
    pos = resting_position(ctx.prop, q, s.to_world(local_xy))
    return BodyInit(
        name=role,
        asset=ctx.prop.key,
        role=role,
        position=pos.tolist(),
        quat=q.tolist(),
        dynamic=ctx.prop.dynamic,
        stencil=stencil,
    )


def _sample_heading(ctx: EventContext) -> float:
    """Surface-local heading. Runs along the long axis fit far more often."""
    if ctx.p("heading_mode") == "uniform":
        return float(ctx.rng.uniform(0, 2 * math.pi))
    base = 0.0 if ctx.surface.size[0] >= ctx.surface.size[1] else math.pi / 2
    if ctx.rng.uniform() < 0.5:
        base += math.pi
    return base + math.radians(ctx.rng.uniform(-1, 1) * ctx.p("long_axis_jitter_deg"))


def decel_estimate(ctx: EventContext) -> float:
    """Rough deceleration of the pushed object on this surface (m/s^2): sliding
    friction for sliders, rolling resistance for rollers. Push calibration
    corrects any residual error in simulation."""
    from .physics.mujoco_backend import combine

    obj, s = ctx.obj, ctx.surface
    if obj.motion == "slide":
        return combine(obj.friction, s.friction, obj.friction_combine) * G
    h = max(obj.shape.support_height(obj.rest_rotation), 1e-3)
    k = 0.4 if obj.shape.type == "sphere" else 0.5  # I / (m r^2)
    return G * obj.rolling_friction / h / (1 + k)


def solve_push(v_event: float, t_event: float, a: float) -> tuple[float, float]:
    """Initial speed and distance so the object moves at ``v_event`` after
    ``t_event`` seconds under constant deceleration ``a``."""
    v0 = v_event + a * t_event
    return v0, v0 * t_event - 0.5 * a * t_event**2


G = 9.81


def _inside(s: Surface, local_xy, margin: float) -> bool:
    return abs(local_xy[0]) <= s.size[0] / 2 - margin and abs(local_xy[1]) <= s.size[1] / 2 - margin


def _straight_run(s: Surface, start, direction, margin: float) -> float:
    """Distance from ``start`` (surface-local) along ``direction`` until the
    point is ``margin`` from an edge."""
    best = math.inf
    for axis in (0, 1):
        d = direction[axis]
        if abs(d) < 1e-9:
            continue
        bound = (s.size[axis] / 2 - margin) * (1 if d > 0 else -1)
        best = min(best, (bound - start[axis]) / d)
    return max(0.0, best)


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #
@register_event("fall")
def fall(ctx: EventContext) -> Layout:
    """Object rolls across the surface and falls off an edge."""
    s = ctx.surface
    if not s.elevated:
        raise ValueError(f"surface {s.name!r} is not elevated; cannot host a fall event")
    r = footprint_radius(ctx.obj)
    margin = r + ctx.p("edge_margin")
    # pick an edge: +x, -x, +y, -y (surface-local outward normals)
    normals = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    idx = int(ctx.rng.integers(4))
    n = np.array(normals[idx], float)
    jitter = math.radians(ctx.rng.uniform(-1, 1) * ctx.p("max_heading_jitter_deg"))
    heading = math.atan2(n[1], n[0]) + jitter
    direction = np.array([math.cos(heading), math.sin(heading)])
    axis = 0 if idx < 2 else 1
    half = s.size[axis] / 2
    t_event = ctx.uniform("event_time_frac") * ctx.clip_duration
    if "distance_to_edge" in ctx.params:
        dist = ctx.uniform("distance_to_edge")
        speed = dist / t_event
    else:
        speed, dist = solve_push(ctx.uniform("edge_speed"), t_event, decel_estimate(ctx))
    # start point: dist from the edge along the path, random along the edge
    along = ctx.rng.uniform(-1, 1) * (s.size[1 - axis] / 2 - margin - 0.05)
    start = np.zeros(2)
    start[axis] = n[axis] * half - direction[axis] * dist
    start[1 - axis] = along - direction[1 - axis] * dist
    if not _inside(s, start, margin):
        raise ValueError("fall start outside surface")
    speed = float(np.clip(speed, ctx.p("min_speed"), ctx.p("max_speed")))
    obj = _make_object(ctx, start, heading, speed)
    edge_local = start + direction * dist
    return Layout(
        event="fall",
        level=ctx.level.name,
        surface=s.name,
        bodies=[obj],
        meta={
            "edge_point": s.to_world(edge_local).tolist(),
            "direction": s.dir_to_world(direction).tolist(),
            "distance_to_edge": dist,
            "speed": speed,
            "expected_event_time": t_event,
        },
    )


@register_event("collision")
def collision(ctx: EventContext) -> Layout:
    """Object rolls into a second object resting on the same surface."""
    s = ctx.surface
    if ctx.prop is None:
        raise ValueError("collision event needs a prop (set events.collision.props or a level override)")
    r_o, r_p = footprint_radius(ctx.obj), footprint_radius(ctx.prop)
    heading = _sample_heading(ctx)
    direction = np.array([math.cos(heading), math.sin(heading)])
    lateral = np.array([-direction[1], direction[0]])
    t_event = ctx.uniform("event_time_frac") * ctx.clip_duration
    if "distance" in ctx.params:
        gap = ctx.uniform("distance")
        speed = gap / t_event
    else:
        speed, gap = solve_push(ctx.uniform("impact_speed"), t_event, decel_estimate(ctx))
    dist = gap + r_o + r_p
    lat = ctx.uniform("lateral_offset_frac") * (r_o + r_p)
    # Midpoint budget: the pair's extent projected on each surface axis, plus
    # room behind the prop for it to be pushed.
    span = dist + 2 * r_p
    extent = np.abs(direction) * span / 2 + max(r_o, r_p) + ctx.p("edge_margin")
    center_budget = np.array([s.size[0] / 2, s.size[1] / 2]) - extent
    if np.any(center_budget <= 0):
        raise ValueError("surface too small for collision layout")
    mid = ctx.rng.uniform(-1, 1, size=2) * center_budget
    start = mid - direction * dist / 2
    prop_xy = mid + direction * dist / 2 + lateral * lat
    margin = ctx.p("edge_margin")
    if not (_inside(s, start, r_o + margin) and _inside(s, prop_xy, r_p + margin)):
        raise ValueError("collision layout outside surface")
    speed = float(np.clip(speed, ctx.p("min_speed"), ctx.p("max_speed")))
    obj = _make_object(ctx, start, heading, speed)
    prop = _make_prop(ctx, prop_xy, "collider", stencil=2)
    return Layout(
        event="collision",
        level=ctx.level.name,
        surface=s.name,
        bodies=[obj, prop],
        meta={
            "direction": s.dir_to_world(direction).tolist(),
            "gap": gap,
            "speed": speed,
            "expected_event_time": t_event,
        },
    )


@register_event("occlusion")
def occlusion(ctx: EventContext) -> Layout:
    """Object rolls behind an occluder and reappears. The camera sampler reads
    ``meta.occluder_side`` to look from the other side of the occluder."""
    s = ctx.surface
    if ctx.prop is None:
        raise ValueError("occlusion event needs an occluder prop")
    r_o, r_p = footprint_radius(ctx.obj), footprint_radius(ctx.prop)
    margin = r_o + ctx.p("edge_margin")
    heading = _sample_heading(ctx)
    direction = np.array([math.cos(heading), math.sin(heading)])
    lateral = np.array([-direction[1], direction[0]])
    side = 1.0 if ctx.rng.uniform() < 0.5 else -1.0
    clearance = ctx.uniform("occluder_clearance")
    lat_off = side * (r_o + r_p + clearance)
    # Start as far back as possible so the whole straight run fits.
    start = ctx.rng.uniform(-0.3, 0.3, size=2) * (np.array(s.size) / 2 - margin)
    back = _straight_run(s, start, -direction, margin)
    start = start - direction * back
    run = _straight_run(s, start, direction, margin)
    min_travel = 2.5 * (r_o + r_p)  # enough to go from visible, to hidden, to visible again
    if "travel_frac" in ctx.params:
        travel = max(ctx.uniform("travel_frac") * run, min(min_travel, run))
        if travel < min_travel:
            raise ValueError("not enough straight run for occlusion")
        occ_frac = ctx.uniform("occluder_position_frac")
        occ_dist = travel * occ_frac
        speed = travel / ctx.clip_duration
        t_cross = occ_frac * ctx.clip_duration
    else:
        a = decel_estimate(ctx)
        t_cross = ctx.uniform("event_time_frac") * ctx.clip_duration
        beyond = 1.3 * (r_o + r_p)  # must travel at least this far past the occluder to reappear
        v_c = ctx.uniform("pass_speed")
        if a > 1e-6:
            v_c = max(v_c, 1.2 * math.sqrt(2 * a * beyond))
        speed, occ_dist = solve_push(v_c, t_cross, a)
        after = v_c * (ctx.clip_duration - t_cross)
        if a > 1e-6:
            after = min(after, v_c**2 / (2 * a))
        travel = occ_dist + max(after, beyond)
        if travel > run:
            raise ValueError("not enough straight run for occlusion")
        slack = run - travel
        start = start + direction * ctx.rng.uniform(0, slack)
    occ_xy = start + direction * occ_dist + lateral * lat_off
    # If the occluder hangs over the edge, slide the whole layout sideways.
    end = start + direction * travel
    for step in np.arange(0.0, 0.6, 0.02):
        shift = -np.sign(lat_off) * lateral * step
        if (_inside(s, occ_xy + shift, r_p + ctx.p("edge_margin"))
                and _inside(s, start + shift, margin) and _inside(s, end + shift, margin)):
            start, occ_xy = start + shift, occ_xy + shift
            break
    else:
        raise ValueError("occluder outside surface")
    speed = float(np.clip(speed, ctx.p("min_speed"), ctx.p("max_speed")))
    obj = _make_object(ctx, start, heading, speed)
    occ = _make_prop(ctx, occ_xy, "occluder", stencil=2)
    return Layout(
        event="occlusion",
        level=ctx.level.name,
        surface=s.name,
        bodies=[obj, occ],
        meta={
            "direction": s.dir_to_world(direction).tolist(),
            "occluder_side": s.dir_to_world(lateral * side).tolist(),
            "travel": travel,
            "speed": speed,
            "expected_event_time": t_cross,
        },
    )
