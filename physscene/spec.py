"""Serializable scene specifications.

Every generated sample is fully described by plain JSON: the layout (initial
conditions), the physical parameters, the simulated trajectory, the camera and
the appearance. This is the "simulator state" that most Unreal-based
benchmarks never release. Keeping it lets anyone regenerate, re-render or
extend a sample, for example with new futures or new viewpoints.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


def _clean(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return [_clean(x) for x in o.tolist()]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(x) for x in o]
    return o


def to_json(o: Any, **kw) -> str:
    if hasattr(o, "__dataclass_fields__"):
        o = asdict(o)
    return json.dumps(_clean(o), **kw)


def stable_id(*parts: Any, n: int = 10) -> str:
    h = hashlib.sha1(json.dumps(_clean(list(parts)), sort_keys=True).encode()).hexdigest()
    return h[:n]


@dataclass
class BodyInit:
    """Initial state of one rigid body.

    ``position`` and ``quat`` give the pose of the **collision primitive**
    (world frame, metres, wxyz). The Unreal actor pose is derived from them
    using the asset's pivot offset."""

    name: str
    asset: str
    role: str  # "object" | "collider" | "occluder"
    position: list[float]
    quat: list[float]
    lin_vel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    ang_vel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    dynamic: bool = True
    stencil: int = 0  # 0 = no mask; 1..7 = mask id


@dataclass
class Layout:
    event: str
    level: str
    surface: str
    bodies: list[BodyInit]
    meta: dict[str, Any] = field(default_factory=dict)

    def body(self, name: str) -> BodyInit:
        for b in self.bodies:
            if b.name == name:
                return b
        raise KeyError(name)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Layout":
        return Layout(
            event=d["event"],
            level=d["level"],
            surface=d["surface"],
            bodies=[BodyInit(**b) for b in d["bodies"]],
            meta=d.get("meta", {}),
        )


@dataclass
class PhysicalParams:
    """Per-body physical parameters (after any perturbation) plus the
    surface/floor contact parameters. Values are absolute, not relative."""

    bodies: dict[str, dict[str, float]]
    surface: dict[str, float]
    floor: dict[str, float]
    gravity: float = 9.81

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "PhysicalParams":
        return PhysicalParams(**d)


@dataclass
class Trajectory:
    """Per-frame body poses sampled at the render frame rate."""

    fps: int
    names: list[str]
    positions: dict[str, list[list[float]]]  # name -> [T][3]
    quats: dict[str, list[list[float]]]  # name -> [T][4]
    contacts: list[list[list[str]]] = field(default_factory=list)  # [T] list of [a, b] pairs
    lin_vel: dict[str, list[list[float]]] = field(default_factory=dict)
    ang_vel_local: dict[str, list[list[float]]] = field(default_factory=dict)  # body-frame omega
    # Per-frame visibility (absent = always visible). Used by "vanish" violations.
    visible: dict[str, list[bool]] = field(default_factory=dict)

    def is_visible(self, name: str, frame: int) -> bool:
        v = self.visible.get(name)
        return True if v is None else bool(v[frame])

    def state_at(self, frame: int) -> dict[str, dict]:
        """Exact per-body state at ``frame`` (for branching a new simulation)."""
        out = {}
        for n in self.names:
            out[n] = {
                "position": self.positions[n][frame],
                "quat": self.quats[n][frame],
                "lin_vel": self.lin_vel[n][frame] if self.lin_vel else [0.0, 0.0, 0.0],
                "ang_vel_local": self.ang_vel_local[n][frame] if self.ang_vel_local else [0.0, 0.0, 0.0],
            }
        return out

    def splice(self, other: "Trajectory", start: int) -> "Trajectory":
        """Frames ``[0, start)`` from self followed by ``other`` (whose frame 0 is frame ``start``)."""
        def cat(a, b):
            return {n: a[n][:start] + b[n] for n in self.names} if a and b else {}

        vis = {}
        if self.visible or other.visible:
            T = start + other.num_frames
            for n in self.names:
                va = self.visible.get(n, [True] * self.num_frames)[:start]
                vb = other.visible.get(n, [True] * other.num_frames)
                vis[n] = (va + vb)[:T]
        return Trajectory(
            fps=self.fps,
            names=list(self.names),
            positions=cat(self.positions, other.positions),
            quats=cat(self.quats, other.quats),
            contacts=self.contacts[:start] + other.contacts,
            lin_vel=cat(self.lin_vel, other.lin_vel),
            ang_vel_local=cat(self.ang_vel_local, other.ang_vel_local),
            visible=vis,
        )

    @property
    def num_frames(self) -> int:
        return len(next(iter(self.positions.values())))

    def pos(self, name: str) -> np.ndarray:
        return np.asarray(self.positions[name], float)

    def quat(self, name: str) -> np.ndarray:
        return np.asarray(self.quats[name], float)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Trajectory":
        return Trajectory(**d)


@dataclass
class CameraSpec:
    """Static pinhole camera. ``rotation`` is camera-to-world (wxyz) in the
    PhysScene frame, where camera local axes are +X forward, +Y left and +Z up
    (Unreal's camera axes, expressed right-handed)."""

    position: list[float]
    quat: list[float]
    hfov_deg: float
    resolution: list[int]
    target: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def intrinsics(self) -> list[list[float]]:
        w, h = self.resolution
        fx = w / (2.0 * np.tan(np.radians(self.hfov_deg) / 2.0))
        return [[fx, 0.0, w / 2.0], [0.0, fx, h / 2.0], [0.0, 0.0, 1.0]]

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "CameraSpec":
        return CameraSpec(**d)


def job_fingerprint(job: dict[str, Any]) -> str:
    """Content hash of a render job; written into the DONE marker so stale renders
    (from an older plan with the same job id) are never reused."""
    return hashlib.sha1(json.dumps(job, sort_keys=True).encode()).hexdigest()[:16]
