"""Physical-validity benchmark construction (reference + matched-prefix candidates)."""
from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any

import numpy as np

from .catalog import Catalog, Level
from .checks import check_physics, visibility
from .raycast import project
from .spec import CameraSpec
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


def view_track(traj: Trajectory, cam: CameraSpec, layout: Layout, level: Level, catalog: Catalog, frames) -> dict[str, np.ndarray]:
    """What the camera sees of the object: unoccluded fraction, image position
    and whether the centre is inside the image, per frame."""
    uv, depth = project(cam, traj.pos("object")[list(frames)])
    w, h = cam.resolution
    inside = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    vis = np.zeros(len(frames))
    for i, t in enumerate(frames):
        if traj.is_visible("object", t) and inside[i]:
            vis[i] = visibility(layout, traj, t, cam, level, catalog, n=24)
    present = np.array([traj.is_visible("object", t) for t in frames])
    return {"vis": vis, "uv": uv, "inside": inside & present}


def visibly_different(ref_tr: dict, cand_tr: dict, px: float, min_frames: int = 3) -> bool:
    """The camera can tell the candidate from the reference in ``min_frames``
    or more frames: visibility flips (appears or disappears), or the visible object
    is ``px`` or more pixels away from where it is in the reference."""
    flip = np.abs(cand_tr["vis"] - ref_tr["vis"]) > 0.5
    both = (cand_tr["vis"] > 0.3) & (ref_tr["vis"] > 0.3)
    moved = both & (np.linalg.norm(cand_tr["uv"] - ref_tr["uv"], axis=1) > px)
    return int((flip | moved).sum()) >= min_frames


def make_candidates(
    cfg: ExperimentConfig,
    level: Level,
    layout: Layout,
    params: PhysicalParams,
    ref: Trajectory,
    key: tuple,
    cams: list[CameraSpec] | None = None,
) -> tuple[list[Candidate], dict[str, Any]]:
    """Reference + valid alternatives + violations for one physics group.

    With ``cams`` given, every candidate must also be visibly different from
    the reference in every view, and (except for ``vanish`` and occlusion events)
    must keep the object in frame for at least ``min_in_frame_frac`` of
    the frames after the prefix. Candidates that change something the camera
    cannot see are dropped."""
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

    T = ref.num_frames
    post = list(range(P + 1, T))
    cams = cams or []
    ref_tracks = [view_track(ref, c, layout, level, cfg.catalog, post) for c in cams]
    px = bm.min_pixel_shift

    def camera_ok(traj: Trajectory, vtype: str | None) -> str:
        """'' if acceptable, else a rejection reason."""
        for cam, rt in zip(cams, ref_tracks):
            ct = view_track(traj, cam, layout, level, cfg.catalog, post)
            if not visibly_different(rt, ct, px * cam.resolution[0] / 1280):
                return "invisible_in_view"
            if vtype != "vanish" and layout.event != "occlusion" and ct["inside"].mean() < bm.min_in_frame_frac:
                return "leaves_frame"
        return ""

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
        # widen the perturbation ranges gradually when draws keep being indistinguishable
        widen = min(2.0, 1.0 + 0.25 * (attempt // max(1, bm.valid_max_attempts // 4)))
        spec = {
            name: type(pert)(rel=min(0.9, pert.rel * widen) if pert.rel is not None else None,
                             abs=pert.abs * widen if pert.abs is not None else None, range=pert.range)
            for name, pert in perturb_spec.items()
        }
        _, fpar, applied = perturb(layout, params, spec, rng)
        traj = branch(ref, P, layout, fpar, sim)
        d = divergence(ref, traj)
        if d < bm.valid_min_divergence:
            stats["valid_rejected_indistinguishable"] += 1
            continue
        if not check_physics(layout, traj, level, cfg.catalog, cfg.checks):
            stats["valid_rejected_event_changed"] += 1
            continue
        reason = camera_ok(traj, None)
        if reason:
            stats[f"valid_rejected_{reason}"] += 1
            continue
        verdict = validate(layout, fpar, traj, level, cfg.catalog, tol)
        if not verdict.valid:
            stats["valid_flagged_by_validator"] += 1
            if bm.require_validator:
                continue
        k += 1
        applied["perturbation_widening"] = widen
        c = Candidate(f"f{k:02d}", "valid", traj, fpar, layout, applied=applied, divergence=d)
        c.applied = {**applied, "validator": verdict.to_dict()}
        out.append(c)
    stats["valid"] = k

    # where the reference object is clearly visible in every view (for vanish onsets)
    if cams:
        seen = np.min(np.stack([rt["vis"] for rt in ref_tracks]), axis=0) >= 0.6
    else:
        seen = np.ones(len(post), bool)

    def vanish_start(frames_invisible: float) -> int | None:
        need = 1 if frames_invisible >= T else min(int(frames_invisible), 4)
        for i in range(1, len(post) - need):
            if seen[i : i + need].all():
                return post[i]
        return None

    def on_support_offset_ok(off) -> bool:
        """A teleport must not move the object off the support it is resting on."""
        tr = ref.pos("object")
        for t in range(P + 1, T):
            p = tr[t]
            for s in level.surfaces.values():
                if s.contains_xy(p) and abs(p[2] - s.top_z) < 0.3:
                    if not s.contains_xy(p + off, margin=0.02):
                        return False
        return True

    # violations
    for vtype, sevs in bm.invalid.items():
        allowed = bm.applicable_events.get(vtype)
        if allowed and layout.event not in allowed:
            continue
        for sev in sevs:
            rng = seeded_rng(cfg.seed, "violation", *key, vtype, sev)
            onset = None
            if vtype == "vanish":
                from .violations import SEVERITIES

                onset = vanish_start(SEVERITIES["vanish"][sev])
                if onset is None:
                    stats["invalid_rejected_invisible_in_view:vanish"] += 1
                    continue
            try:
                traj = make_violation(vtype, sev, ref, P, layout, params, sim, rng, vanish_start=onset,
                                      offset_ok=on_support_offset_ok)
            except ValueError:
                stats[f"invalid_rejected_no_clean_construction:{vtype}"] += 1
                continue
            d = divergence(ref, traj)
            if d < bm.invalid_min_divergence:
                stats[f"invalid_rejected_indistinguishable:{vtype}"] += 1
                continue
            reason = camera_ok(traj, vtype)
            if reason:
                stats[f"invalid_rejected_{reason}:{vtype}"] += 1
                continue
            verdict = validate(layout, params, traj, level, cfg.catalog, tol)
            if verdict.valid:
                stats[f"invalid_missed_by_validator:{vtype}"] += 1
                if bm.require_validator:
                    continue
            else:
                stats[f"invalid_confirmed:{vtype}"] += 1
            c = Candidate(f"{vtype}_{sev}", "invalid", traj, params, layout, violation=violation_meta(vtype, sev, P, onset), divergence=d)
            c.applied = {"validator": verdict.to_dict()}
            out.append(c)
    stats["invalid"] = sum(1 for c in out if c.kind == "invalid")
    return out, dict(stats)


def candidate_id(cfg: ExperimentConfig, group_id: str, cid: str) -> str:
    return stable_id(cfg.name, cfg.seed, group_id, cid)
