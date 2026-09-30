"""Candidate futures for physical-validity benchmarks.

Every candidate shares a **matched prefix** with its reference: frames
``0..P`` are bit-identical (copied), and the continuation starts from the
exact simulator state at frame ``P``. Candidates are either

* **valid** alternatives: the simulation is branched at ``P`` with resampled
  *hidden* parameters (friction, restitution, mass, rolling friction,
  surface friction). No state is edited, so the future is physically
  plausible but different; or
* **invalid** violations: a controlled breach of physical law after ``P``,
  with a type and severity. Some are dynamic (re-simulated from ``P`` under
  broken physics) and some are kinematic (the recorded trajectory is edited).

The ground-truth label comes from how the candidate was constructed, never
from a metric. :mod:`physscene.validator` independently re-checks every label
from the saved states.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .spec import Layout, PhysicalParams, Trajectory

FAR_AWAY = [0.0, 0.0, -1000.0]  # where "vanished" bodies are parked (never rendered in frame)

# Default severities per violation type (the parameter each severity maps to).
SEVERITIES: dict[str, dict[str, float]] = {
    "teleport": {"low": 0.05, "mid": 0.15, "high": 0.3},  # jump distance (m)
    "speed_jump": {"low": 1.5, "mid": 2.5, "high": 4.0},  # velocity multiplier at P
    "gravity": {"low": 0.5, "mid": 0.1, "high": -0.5},  # gravity multiplier after P
    "freeze": {"mid": 1.0},  # object stops dead
    "time_reversal": {"mid": 1.0},  # motion plays backwards after P
    "vanish": {"low": 8.0, "mid": 16.0, "high": 1e9},  # frames invisible (high = never returns)
    "penetration": {"mid": 1.0},  # contact with the next support/collider disabled
}


@dataclass
class Candidate:
    cid: str  # e.g. "ref", "f01", "teleport_mid"
    kind: str  # "reference" | "valid" | "invalid"
    traj: Trajectory
    params: PhysicalParams
    layout: Layout
    violation: dict[str, Any] | None = None
    applied: dict[str, float] = field(default_factory=dict)
    divergence: float = 0.0  # max object position deviation from the reference (m)

    @property
    def label(self) -> str:
        return "invalid" if self.kind == "invalid" else "valid"


SimFn = Callable[..., Trajectory]  # (layout, params, init_state=..., exclude_pairs=..., num_frames=...) -> Trajectory


def divergence(ref: Trajectory, cand: Trajectory, body: str | None = None) -> float:
    """Max position deviation from the reference (m), over ``body`` or over all bodies."""
    out = 0.0
    for n in [body] if body else ref.names:
        a, b = ref.pos(n), cand.pos(n)
        vis = np.array([cand.is_visible(n, t) for t in range(cand.num_frames)])
        d = np.linalg.norm(a - b, axis=1)
        d[~vis] = 1.0  # vanishing counts as maximal deviation
        out = max(out, float(d.max()))
    return out


def branch(ref: Trajectory, P: int, layout: Layout, params: PhysicalParams, sim: SimFn, state_edit=None, **kw) -> Trajectory:
    """Re-simulate from the exact state at frame ``P`` and splice onto ``ref[:P]``."""
    state = ref.state_at(P)
    if state_edit:
        state = state_edit(copy.deepcopy(state))
    cont = sim(layout, params, init_state=state, num_frames=ref.num_frames - P, **kw)
    return ref.splice(cont, P)


# --------------------------------------------------------------------------- #
# Kinematic edits
# --------------------------------------------------------------------------- #
def _copy(t: Trajectory) -> Trajectory:
    return Trajectory.from_dict(copy.deepcopy(t.__dict__))


def teleport(ref: Trajectory, P: int, dist: float, rng: np.random.Generator, body="object", delay=3) -> Trajectory:
    t = _copy(ref)
    ang = rng.uniform(0, 2 * math.pi)
    off = np.array([math.cos(ang), math.sin(ang), 0.0]) * dist
    for f in range(min(P + delay, t.num_frames), t.num_frames):
        t.positions[body][f] = (np.asarray(t.positions[body][f]) + off).tolist()
    return t


def freeze(ref: Trajectory, P: int, body="object") -> Trajectory:
    t = _copy(ref)
    for f in range(P + 1, t.num_frames):
        t.positions[body][f] = list(t.positions[body][P])
        t.quats[body][f] = list(t.quats[body][P])
    return t


def time_reversal(ref: Trajectory, P: int, body="object") -> Trajectory:
    t = _copy(ref)
    for f in range(P + 1, t.num_frames):
        src = max(0, 2 * P - f)
        t.positions[body][f] = list(ref.positions[body][src])
        t.quats[body][f] = list(ref.quats[body][src])
    return t


def vanish(ref: Trajectory, P: int, frames: float, body="object", delay=2) -> Trajectory:
    t = _copy(ref)
    vis = [True] * t.num_frames
    start = min(P + delay, t.num_frames - 1)
    end = t.num_frames if frames >= t.num_frames else min(t.num_frames, start + int(frames))
    for f in range(start, end):
        vis[f] = False
    t.visible = {n: [True] * t.num_frames for n in t.names}
    t.visible[body] = vis
    return t


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
def make_violation(
    vtype: str,
    severity: str,
    ref: Trajectory,
    P: int,
    layout: Layout,
    params: PhysicalParams,
    sim: SimFn,
    rng: np.random.Generator,
) -> Trajectory:
    value = SEVERITIES[vtype][severity]
    if vtype == "teleport":
        return teleport(ref, P, value, rng)
    if vtype == "freeze":
        return freeze(ref, P)
    if vtype == "time_reversal":
        return time_reversal(ref, P)
    if vtype == "vanish":
        return vanish(ref, P, value)
    if vtype == "speed_jump":
        def edit(state):
            s = state["object"]
            s["lin_vel"] = (np.asarray(s["lin_vel"]) * value).tolist()
            s["ang_vel_local"] = (np.asarray(s["ang_vel_local"]) * value).tolist()
            return state

        return branch(ref, P, layout, params, sim, state_edit=edit)
    if vtype == "gravity":
        p2 = PhysicalParams.from_dict(copy.deepcopy(params.__dict__))
        p2.gravity = params.gravity * value
        return branch(ref, P, layout, p2, sim)
    if vtype == "penetration":
        partner = "collider" if layout.event == "collision" else "surface"
        return branch(ref, P, layout, params, sim, exclude_pairs={frozenset(("object", partner))})
    raise ValueError(f"unknown violation type {vtype!r}")


def violation_meta(vtype: str, severity: str, P: int) -> dict[str, Any]:
    return {"type": vtype, "severity": severity, "value": SEVERITIES[vtype][severity], "onset_frame": P}
