"""Build primitive lists for a given frame (for visibility checks and preview rendering)."""
from __future__ import annotations

import math

import numpy as np

from .catalog import Catalog, Level
from .geometry import quat_from_yaw
from .raycast import Primitive
from .spec import Layout, Trajectory

SURFACE_COLOR = (0.62, 0.45, 0.3)
COLLIDER_COLOR = (0.5, 0.5, 0.55)


def scene_primitives(
    layout: Layout,
    traj: Trajectory,
    frame: int,
    level: Level,
    catalog: Catalog,
    colors: dict[str, tuple[float, float, float]] | None = None,
) -> list[Primitive]:
    colors = colors or {}
    prims = [Primitive("floor", None, np.array([0, 0, level.floor_z]), np.array([1, 0, 0, 0]), level.preview_floor_color)]
    for s in level.surfaces.values():
        shape, c, q = s.box_shape()
        prims.append(Primitive(f"surface:{s.name}", shape, c, q, SURFACE_COLOR))
    for c in level.colliders:
        prims.append(
            Primitive(
                f"static:{c.name}",
                c.shape,
                np.asarray(c.center, float),
                quat_from_yaw(math.radians(c.yaw_deg)),
                COLLIDER_COLOR,
            )
        )
    for b in layout.bodies:
        a = catalog.get(b.asset)
        prims.append(
            Primitive(
                b.name,
                a.shape,
                np.asarray(traj.positions[b.name][frame], float),
                np.asarray(traj.quats[b.name][frame], float),
                colors.get(b.name, (0.8, 0.8, 0.8)),
                stencil=b.stencil,
            )
        )
    return prims
