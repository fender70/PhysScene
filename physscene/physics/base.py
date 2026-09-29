from __future__ import annotations

from abc import ABC, abstractmethod

from ..catalog import Catalog, Level
from ..spec import Layout, PhysicalParams, Trajectory


def default_params(layout: Layout, level: Level, catalog: Catalog) -> PhysicalParams:
    """Unperturbed physical parameters taken from the catalog and level."""
    bodies = {}
    for b in layout.bodies:
        a = catalog.get(b.asset)
        bodies[b.name] = {
            "mass": a.mass,
            "friction": a.friction,
            "restitution": a.restitution,
            "rolling_friction": a.rolling_friction,
            "torsional_friction": a.torsional_friction,
            "linear_damping": a.linear_damping,
            "angular_damping": a.angular_damping,
            "friction_combine": a.friction_combine,
            "restitution_combine": a.restitution_combine,
        }
    s = level.surfaces[layout.surface]
    return PhysicalParams(
        bodies=bodies,
        surface={"friction": s.friction, "restitution": s.restitution},
        floor={"friction": level.floor_friction, "restitution": level.floor_restitution},
    )


class PhysicsBackend(ABC):
    name: str = "base"

    @abstractmethod
    def simulate(
        self,
        layout: Layout,
        params: PhysicalParams,
        level: Level,
        catalog: Catalog,
        fps: int,
        num_frames: int,
        timestep: float = 0.001,
        pre_roll: float = 0.0,
    ) -> Trajectory:
        """Simulate and return poses sampled at ``fps`` for ``num_frames`` frames.
        Frame 0 is the state at ``t = pre_roll``."""
