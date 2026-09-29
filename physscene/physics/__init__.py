"""Physics backends. Each one turns a Layout + PhysicalParams into a Trajectory."""
from __future__ import annotations

from .base import PhysicsBackend, default_params


def get_backend(name: str) -> PhysicsBackend:
    if name == "mujoco":
        from .mujoco_backend import MujocoBackend

        return MujocoBackend()
    raise ValueError(f"unknown physics backend {name!r} (available: mujoco)")


__all__ = ["PhysicsBackend", "default_params", "get_backend"]
