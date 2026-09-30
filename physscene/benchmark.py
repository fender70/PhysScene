"""Physical-validity benchmark construction (reference + matched-prefix candidates)."""
from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any

from .catalog import Catalog, Level
from .checks import check_physics
from .config import ExperimentConfig
from .physics import get_backend
from .spec import Layout, PhysicalParams, Trajectory, stable_id, to_json
from .validator import validate_states
from .violations import Candidate, branch, divergence, make_violation, violation_meta

log = logging.getLogger("physscene.benchmark")

HIDDEN_PREFIXES = ("initial_",)  # velocity edits are not hidden parameters


def world_geometry(level: Level) -> dict[str, Any]:
    return {
        "floor_z": level.floor_z,
        "supports": [
            {"name": s.name, "center": list(s.center), "size": list(s.size), "yaw_deg": s.yaw_deg} for s in level.surfaces.values()
        ],
    }


def body_shapes(layout: Layout, catalog: Catalog) -> dict[str, dict]:
    return {b.name: catalog.get(b.asset).shape.to_dict() for b in layout.bodies}


def validate(layout: Layout, params: PhysicalParams, traj: Trajectory, level: Level, catalog: Catalog, tol=None):
    w = world_geometry(level)
    states = {"layout": json.loads(to_json(layout)), "params": json.loads(to_json(params)), "trajectory": json.loads(to_json(traj))}
    return validate_states(states, body_shapes(layout, catalog), w["supports"], w["floor_z"], tol)


def make_candidates(
    cfg: ExperimentConfig,
    level: Level,
    layout: Layout,
    params: PhysicalParams,
    ref: Trajectory,
    key: tuple,
) -> tuple[list[Candidate], dict[str, Any]]:
    """Reference + valid alternatives + violations for one physics group."""
    from .planner import perturb, seeded_rng  # local import (planner imports this module)

    bm = cfg.benchmark
    P = bm.prefix_frames
    backend = get_backend(cfg.physics.backend)
    tol = bm.validator_tol
    stats: Counter = Counter()

    def sim(lay, par, init_state=None, num_frames=None, exclude_pairs=None):
        return backend.simulate(
            lay, par, level, cfg.catalog, cfg.render.fps, num_frames or cfg.render.num_frames,
            cfg.physics.timestep, 0.0, init_state, exclude_pairs,
        )

    out = []
    ref_verdict = validate(layout, params, ref, level, cfg.catalog, tol)
    out.append(Candidate("ref", "reference", ref, params, layout))
    out[-1].applied = {"validator": ref_verdict.to_dict()}
    if not ref_verdict.valid:
        stats["reference_flagged"] += 1
        log.warning("%s: reference flagged by validator: %s", key, ref_verdict.summary)

    # valid alternatives: branch at P with resampled hidden parameters
    perturb_spec = {k: v for k, v in bm.valid_perturb.items() if not k.startswith(HIDDEN_PREFIXES)}
    k = 0
    for attempt in range(bm.valid_count * bm.valid_max_attempts):
        if k >= bm.valid_count:
            break
        rng = seeded_rng(cfg.seed, "valid", *key, attempt)
        _, fpar, applied = perturb(layout, params, perturb_spec, rng)
        traj = branch(ref, P, layout, fpar, sim)
        d = divergence(ref, traj)
        if d < bm.valid_min_divergence:
            stats["valid_rejected_indistinguishable"] += 1
            continue
        if not check_physics(layout, traj, level, cfg.catalog, cfg.checks):
            stats["valid_rejected_event_changed"] += 1
            continue
        verdict = validate(layout, fpar, traj, level, cfg.catalog, tol)
        if not verdict.valid:
            stats["valid_flagged_by_validator"] += 1
            if bm.require_validator:
                continue
        k += 1
        c = Candidate(f"f{k:02d}", "valid", traj, fpar, layout, applied=applied, divergence=d)
        c.applied = {**applied, "validator": verdict.to_dict()}
        out.append(c)
    stats["valid"] = k

    # violations
    for vtype, sevs in bm.invalid.items():
        allowed = bm.applicable_events.get(vtype)
        if allowed and layout.event not in allowed:
            continue
        for sev in sevs:
            rng = seeded_rng(cfg.seed, "violation", *key, vtype, sev)
            traj = make_violation(vtype, sev, ref, P, layout, params, sim, rng)
            d = divergence(ref, traj)
            if d < bm.invalid_min_divergence:
                stats[f"invalid_rejected_indistinguishable:{vtype}"] += 1
                continue
            verdict = validate(layout, params, traj, level, cfg.catalog, tol)
            if verdict.valid:
                stats[f"invalid_missed_by_validator:{vtype}"] += 1
                if bm.require_validator:
                    continue
            else:
                stats[f"invalid_confirmed:{vtype}"] += 1
            c = Candidate(f"{vtype}_{sev}", "invalid", traj, params, layout, violation=violation_meta(vtype, sev, P), divergence=d)
            c.applied = {"validator": verdict.to_dict()}
            out.append(c)
    stats["invalid"] = sum(1 for c in out if c.kind == "invalid")
    return out, dict(stats)


def candidate_id(cfg: ExperimentConfig, group_id: str, cid: str) -> str:
    return stable_id(cfg.name, cfg.seed, group_id, cid)
