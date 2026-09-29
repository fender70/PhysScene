"""Asset catalog and level manifests.

A researcher describes each asset and each level **once**. The planner then
generates arbitrarily many scenes from them, with no per-scene manual work.

* **Asset catalog** (``catalog.yaml``): for each object and prop, the Unreal
  mesh, the material per appearance, a simple collision primitive and physical
  properties.
* **Level manifest** (``levels/<name>.yaml``): the Unreal map, the support
  surfaces events can happen on, the floor height, static colliders and camera
  constraints. It can be written by hand or exported automatically from
  actors tagged ``PhysSceneSurface`` (see ``unreal/physscene_ue/level_export.py``).

All lengths are metres in the right-handed PhysScene frame (see
:mod:`physscene.geometry`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .geometry import quat_from_yaw, quat_identity, quat_to_matrix


# --------------------------------------------------------------------------- #
# Shapes
# --------------------------------------------------------------------------- #
@dataclass
class Shape:
    """Collision / proxy primitive in the body's local frame.

    ``sphere``: ``radius``; ``box``: ``half_extents`` (x, y, z);
    ``cylinder`` and ``capsule``: ``radius`` and ``half_length`` along local Z.
    """

    type: str
    radius: float = 0.0
    half_extents: tuple[float, float, float] = (0.0, 0.0, 0.0)
    half_length: float = 0.0

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Shape":
        t = d["type"]
        if t == "sphere":
            return Shape(t, radius=float(d["radius"]))
        if t == "box":
            he = d.get("half_extents")
            if he is None:
                he = [s / 2 for s in d["size"]]
            return Shape(t, half_extents=tuple(float(x) for x in he))
        if t in ("cylinder", "capsule"):
            hl = d.get("half_length", d.get("length", 0) / 2)
            return Shape(t, radius=float(d["radius"]), half_length=float(hl))
        raise ValueError(f"unknown shape type {t!r}")

    def to_dict(self) -> dict[str, Any]:
        if self.type == "sphere":
            return {"type": "sphere", "radius": self.radius}
        if self.type == "box":
            return {"type": "box", "half_extents": list(self.half_extents)}
        return {"type": self.type, "radius": self.radius, "half_length": self.half_length}

    def local_half_extents(self) -> np.ndarray:
        if self.type == "sphere":
            return np.full(3, self.radius)
        if self.type == "box":
            return np.asarray(self.half_extents, float)
        extra = self.radius if self.type == "capsule" else 0.0
        return np.array([self.radius, self.radius, self.half_length + extra])

    def bounding_radius(self) -> float:
        return float(np.linalg.norm(self.local_half_extents())) if self.type != "sphere" else self.radius

    def support_height(self, quat) -> float:
        """Distance from the body origin to its lowest point along world -Z."""
        if self.type == "sphere":
            return self.radius
        r = quat_to_matrix(quat)
        down = r.T @ np.array([0.0, 0.0, -1.0])  # world down expressed locally
        if self.type == "box":
            return float(np.sum(np.abs(down) * np.asarray(self.half_extents)))
        # cylinder / capsule: axis along local z
        axial = abs(down[2]) * self.half_length
        radial = math.sqrt(max(0.0, 1.0 - down[2] ** 2)) * self.radius
        if self.type == "capsule":
            return axial + self.radius
        return axial + radial


# --------------------------------------------------------------------------- #
# Assets
# --------------------------------------------------------------------------- #
@dataclass
class Appearance:
    name: str
    materials: dict[int, str] = field(default_factory=dict)  # slot index -> UE material path
    preview_color: tuple[float, float, float] = (0.7, 0.7, 0.7)
    prompt: str | None = None  # e.g. "orange" for VLM prompts


@dataclass
class Asset:
    """An object that can be pushed (``role: object``) or a prop used as a
    collider / occluder (``role: prop``)."""

    key: str
    prompt_name: str
    ue_mesh: str
    shape: Shape
    mass: float = 0.1
    friction: float = 0.6
    restitution: float = 0.3
    rolling_friction: float = 0.002
    torsional_friction: float = 0.005
    linear_damping: float = 0.0
    angular_damping: float = 0.0
    # Unreal-style combine modes (average | min | multiply | max). For a pair the
    # "higher" mode wins, in the same order as Unreal's EFrictionCombineMode.
    friction_combine: str = "average"
    restitution_combine: str = "average"
    # Pose of the collision primitive relative to the mesh pivot (mesh-local, metres / wxyz).
    pivot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pivot_rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # How the object moves when pushed. ``roll`` gives it matching spin (no-slip);
    # ``slide`` gives it only linear velocity (e.g. a toy car on low friction).
    motion: str = "roll"
    # Resting orientation (wxyz) applied before yaw. For a can lying on its side,
    # rotate local Z into the horizontal plane.
    rest_rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    # Local axis (after rest_rotation) that should point along the push
    # direction. For rolling cylinders this is perpendicular to the cylinder axis.
    forward_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    dynamic: bool = True  # props: can they be knocked over?
    appearances: dict[str, Appearance] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    @staticmethod
    def from_dict(key: str, d: dict[str, Any]) -> "Asset":
        apps = {}
        for name, a in (d.get("appearances") or {"default": {}}).items():
            a = a or {}
            mats = a.get("materials", {})
            if "material" in a:
                mats = {0: a["material"], **mats}
            apps[name] = Appearance(
                name=name,
                materials={int(k): v for k, v in mats.items()},
                preview_color=tuple(a.get("preview_color", (0.7, 0.7, 0.7))),
                prompt=a.get("prompt", name),
            )
        return Asset(
            key=key,
            prompt_name=d.get("prompt_name", key),
            ue_mesh=d.get("ue_mesh", ""),
            shape=Shape.from_dict(d["shape"]),
            mass=float(d.get("mass", 0.1)),
            friction=float(d.get("friction", 0.6)),
            restitution=float(d.get("restitution", 0.3)),
            rolling_friction=float(d.get("rolling_friction", 0.002)),
            torsional_friction=float(d.get("torsional_friction", 0.005)),
            linear_damping=float(d.get("linear_damping", 0.0)),
            angular_damping=float(d.get("angular_damping", 0.0)),
            friction_combine=d.get("friction_combine", "average"),
            restitution_combine=d.get("restitution_combine", "average"),
            pivot_offset=tuple(d.get("pivot_offset", (0, 0, 0))),
            pivot_rotation=tuple(d.get("pivot_rotation", (1, 0, 0, 0))),
            scale=tuple(d.get("scale", (1, 1, 1))),
            motion=d.get("motion", "roll"),
            rest_rotation=tuple(d.get("rest_rotation", (1, 0, 0, 0))),
            forward_axis=tuple(d.get("forward_axis", (1, 0, 0))),
            dynamic=bool(d.get("dynamic", True)),
            appearances=apps,
            tags=list(d.get("tags", [])),
        )


@dataclass
class Catalog:
    objects: dict[str, Asset]
    props: dict[str, Asset]

    def get(self, key: str) -> Asset:
        if key in self.objects:
            return self.objects[key]
        if key in self.props:
            return self.props[key]
        raise KeyError(f"asset {key!r} not in catalog")

    @staticmethod
    def load(path: str | Path) -> "Catalog":
        d = yaml.safe_load(Path(path).read_text())
        return Catalog(
            objects={k: Asset.from_dict(k, v) for k, v in (d.get("objects") or {}).items()},
            props={k: Asset.from_dict(k, v) for k, v in (d.get("props") or {}).items()},
        )


# --------------------------------------------------------------------------- #
# Levels
# --------------------------------------------------------------------------- #
@dataclass
class Surface:
    """A horizontal rectangular support surface, e.g. a table top or a floor
    patch. ``center`` is the centre of the **top face**."""

    name: str
    center: tuple[float, float, float]
    size: tuple[float, float]
    yaw_deg: float = 0.0
    thickness: float = 0.04
    friction: float = 0.5
    restitution: float = 0.2
    prompt_name: str = "table"
    elevated: bool = True  # can objects fall off it?
    ue_material: Any = None  # used when the level has build_geometry: true

    @property
    def top_z(self) -> float:
        return float(self.center[2])

    def rotation(self):
        return quat_from_yaw(math.radians(self.yaw_deg))

    def to_world(self, local_xy) -> np.ndarray:
        """Surface-local (x, y) on the top face -> world xyz."""
        c, s = math.cos(math.radians(self.yaw_deg)), math.sin(math.radians(self.yaw_deg))
        x, y = local_xy
        return np.array([self.center[0] + c * x - s * y, self.center[1] + s * x + c * y, self.top_z])

    def dir_to_world(self, local_dir) -> np.ndarray:
        c, s = math.cos(math.radians(self.yaw_deg)), math.sin(math.radians(self.yaw_deg))
        x, y = local_dir
        return np.array([c * x - s * y, s * x + c * y, 0.0])

    def contains_xy(self, p, margin: float = 0.0) -> bool:
        c, s = math.cos(math.radians(self.yaw_deg)), math.sin(math.radians(self.yaw_deg))
        dx, dy = p[0] - self.center[0], p[1] - self.center[1]
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        return abs(lx) <= self.size[0] / 2 - margin and abs(ly) <= self.size[1] / 2 - margin

    def box_shape(self) -> tuple[Shape, np.ndarray, np.ndarray]:
        """(shape, centre, quat) of this surface as a solid box."""
        he = (self.size[0] / 2, self.size[1] / 2, self.thickness / 2)
        center = np.array([self.center[0], self.center[1], self.top_z - self.thickness / 2])
        return Shape("box", half_extents=he), center, self.rotation()


@dataclass
class StaticCollider:
    name: str
    shape: Shape
    center: tuple[float, float, float]
    yaw_deg: float = 0.0
    ue_material: Any = None


@dataclass
class CameraConstraints:
    distance: tuple[float, float] = (0.8, 3.0)
    elevation_deg: tuple[float, float] = (10.0, 40.0)
    hfov_deg: tuple[float, float] = (50.0, 50.0)
    min_height: float = 0.2
    # Axis-aligned world box the camera must stay inside, e.g. room walls.
    bounds_min: tuple[float, float, float] | None = None
    bounds_max: tuple[float, float, float] | None = None


@dataclass
class Level:
    name: str
    ue_map: str
    surfaces: dict[str, Surface]
    floor_z: float = 0.0
    floor_friction: float = 0.6
    floor_restitution: float = 0.2
    colliders: list[StaticCollider] = field(default_factory=list)
    camera: CameraConstraints = field(default_factory=CameraConstraints)
    # Event-specific prop and surface choices, e.g. {"collision": {"prop": "CeramicCup"}}.
    event_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    hide_actors: list[str] = field(default_factory=list)  # UE actor labels to hide when rendering
    # Spawn surfaces/colliders as meshes in Unreal (for procedural / blank levels).
    build_geometry: bool = False
    preview_floor_color: tuple[float, float, float] = (0.55, 0.5, 0.45)

    @staticmethod
    def load(path: str | Path) -> "Level":
        d = yaml.safe_load(Path(path).read_text())
        surfaces = {}
        for name, s in d["surfaces"].items():
            surfaces[name] = Surface(
                name=name,
                center=tuple(s["center"]),
                size=tuple(s["size"]),
                yaw_deg=float(s.get("yaw_deg", 0.0)),
                thickness=float(s.get("thickness", 0.04)),
                friction=float(s.get("friction", 0.5)),
                restitution=float(s.get("restitution", 0.2)),
                prompt_name=s.get("prompt_name", name),
                elevated=bool(s.get("elevated", s["center"][2] - d.get("floor_z", 0.0) > 0.15)),
                ue_material=s.get("ue_material"),
            )
        colliders = [
            StaticCollider(
                c["name"], Shape.from_dict(c["shape"]), tuple(c["center"]), float(c.get("yaw_deg", 0)), c.get("ue_material")
            )
            for c in d.get("colliders", [])
        ]
        cam = d.get("camera", {})
        cc = CameraConstraints(
            distance=tuple(cam.get("distance", (0.8, 3.0))),
            elevation_deg=tuple(cam.get("elevation_deg", (10.0, 40.0))),
            hfov_deg=tuple(cam.get("hfov_deg", (50.0, 50.0))),
            min_height=float(cam.get("min_height", 0.2)),
            bounds_min=tuple(cam["bounds_min"]) if "bounds_min" in cam else None,
            bounds_max=tuple(cam["bounds_max"]) if "bounds_max" in cam else None,
        )
        return Level(
            name=d["name"],
            ue_map=d.get("ue_map", ""),
            surfaces=surfaces,
            floor_z=float(d.get("floor_z", 0.0)),
            floor_friction=float(d.get("floor_friction", 0.6)),
            floor_restitution=float(d.get("floor_restitution", 0.2)),
            colliders=colliders,
            camera=cc,
            event_overrides=d.get("events", {}) or {},
            hide_actors=list(d.get("hide_actors", [])),
            build_geometry=bool(d.get("build_geometry", False)),
            preview_floor_color=tuple(d.get("preview_floor_color", (0.55, 0.5, 0.45))),
        )


__all__ = [
    "Shape",
    "Appearance",
    "Asset",
    "Catalog",
    "Surface",
    "StaticCollider",
    "CameraConstraints",
    "Level",
    "quat_identity",
]
