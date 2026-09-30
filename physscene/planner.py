"""Planner: experiment config -> validated, fully specified render jobs.

Pipeline for every *physics group* (event x scene x object):

1. propose a layout with the event template (seeded, so the same attempt looks
   alike across objects);
2. simulate it with the physics backend;
3. verify the event with :func:`physscene.checks.check_physics`;
4. sample ``views`` cameras and verify each with :func:`check_view`;
5. optionally sample alternative futures (perturbed hidden physics, identical frame 0);
6. write one render job per (appearance x view x future).

Every step is seeded from the experiment seed, so the same config always
produces the same plan.
"""
from __future__ import annotations

import json
import logging
import math
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from .camera import sample_camera
from .catalog import Asset, Level
from .checks import check_physics, check_view
from .config import ExperimentConfig, Perturbation
from .events import EVENT_REGISTRY, EventContext, event_params
from .geometry import quat_from_yaw, quat_mul, quat_to_ue_rotator, rotate, to_ue_location, unwrap_degrees
from .physics import default_params, get_backend
from .violations import FAR_AWAY
from .benchmark import world_geometry
from .provenance import provenance_info
from .spec import CameraSpec, Layout, PhysicalParams, Trajectory, stable_id, to_json

log = logging.getLogger("physscene.planner")


def seeded_rng(*parts: Any) -> np.random.Generator:
    return np.random.default_rng(int(stable_id(*parts, n=15), 16))


# --------------------------------------------------------------------------- #
# Selection helpers
# --------------------------------------------------------------------------- #
def choose_surface(level: Level, event: str) -> str:
    ov = level.event_overrides.get(event, {})
    if "surface" in ov:
        return ov["surface"]
    for name, s in level.surfaces.items():
        if event != "fall" or s.elevated:
            return name
    raise ValueError(f"level {level.name!r} has no surface suitable for {event!r}")


def choose_prop(cfg: ExperimentConfig, level: Level, event: str, rng: np.random.Generator) -> Asset | None:
    if event not in ("collision", "occlusion") and "prop" not in level.event_overrides.get(event, {}):
        return None
    ov = level.event_overrides.get(event, {})
    if "prop" in ov:
        return cfg.catalog.get(ov["prop"])
    props = (cfg.events.get(event) or {}).get("props") or [
        k for k, a in cfg.catalog.props.items() if event in a.tags or not a.tags
    ]
    if not props:
        raise ValueError(f"no props available for event {event!r}")
    return cfg.catalog.get(props[int(rng.integers(len(props)))])


# --------------------------------------------------------------------------- #
# Futures
# --------------------------------------------------------------------------- #
def _perturb_value(v: float, pert: Perturbation, rng: np.random.Generator) -> float:
    if pert.range is not None:
        return float(rng.uniform(*pert.range))
    if pert.rel is not None:
        v = v * float(rng.uniform(1 - pert.rel, 1 + pert.rel))
    if pert.abs is not None:
        v = v + float(rng.uniform(-pert.abs, pert.abs))
    return v


def perturb(
    layout: Layout, params: PhysicalParams, spec: dict[str, Perturbation], rng: np.random.Generator
) -> tuple[Layout, PhysicalParams, dict[str, float]]:
    """Resample hidden physical parameters. Positions and orientations at
    frame 0 are never touched, so every future starts from the same image.

    Keys: ``<param>`` (object body), ``<body>.<param>``, ``surface.<param>``,
    ``floor.<param>``, ``initial_speed`` and ``initial_direction_deg``."""
    lay = Layout.from_dict(json.loads(to_json(layout)))
    par = PhysicalParams.from_dict(json.loads(to_json(params)))
    applied: dict[str, float] = {}
    for key, pert in sorted(spec.items()):
        if key == "initial_speed":
            f = _perturb_value(1.0, pert, rng)
            b = lay.body("object")
            b.lin_vel = (np.asarray(b.lin_vel) * f).tolist()
            b.ang_vel = (np.asarray(b.ang_vel) * f).tolist()
            applied[key] = f
            continue
        if key == "initial_direction_deg":
            a = math.radians(_perturb_value(0.0, pert, rng))
            q = quat_from_yaw(a)
            b = lay.body("object")
            b.lin_vel = rotate(q, b.lin_vel).tolist()
            b.ang_vel = rotate(q, b.ang_vel).tolist()
            applied[key] = math.degrees(a)
            continue
        target, _, name = key.rpartition(".")
        target = target or "object"
        if target == "surface":
            d = par.surface
        elif target == "floor":
            d = par.floor
        elif target == "gravity":
            par.gravity = _perturb_value(par.gravity, pert, rng)
            applied[key] = par.gravity
            continue
        else:
            if target not in par.bodies:
                continue
            d = par.bodies[target]
        if name not in d:
            raise KeyError(f"unknown physical parameter {key!r}")
        d[name] = max(0.0, _perturb_value(d[name], pert, rng))
        if name == "restitution":
            d[name] = min(d[name], 0.98)
        applied[key] = d[name]
    return lay, par, applied


# --------------------------------------------------------------------------- #
# UE export of a trajectory
# --------------------------------------------------------------------------- #
def actor_keys(asset: Asset, traj: Trajectory, name: str) -> dict[str, list[list[float]]]:
    """Convert primitive poses to Unreal actor pivot keys (cm, roll/pitch/yaw deg)."""
    piv_q = np.asarray(asset.pivot_rotation, float)
    piv_q_inv = piv_q * np.array([1, -1, -1, -1])
    piv_off = np.asarray(asset.pivot_offset, float)
    locs, rots = [], []
    for f, (p, q) in enumerate(zip(traj.positions[name], traj.quats[name])):
        qa = quat_mul(q, piv_q_inv)
        pa = np.asarray(p) - rotate(qa, piv_off)
        if not traj.is_visible(name, f):
            pa = np.asarray(FAR_AWAY)  # hidden: parked far below the level
        locs.append([round(v, 4) for v in to_ue_location(pa)])
        rots.append(quat_to_ue_rotator(qa))
    rots = [[round(v, 4) for v in r] for r in unwrap_degrees(rots)]
    return {"location": locs, "rotation": rots}


def camera_ue(cam: CameraSpec) -> dict[str, Any]:
    return {
        "location": to_ue_location(cam.position),
        "rotation": quat_to_ue_rotator(cam.quat),
        "hfov_deg": cam.hfov_deg,
    }


# --------------------------------------------------------------------------- #
# Groups
# --------------------------------------------------------------------------- #
@dataclass
class GroupResult:
    group_id: str
    event: str
    scene: str
    obj: str
    ok: bool
    reason: str = ""
    attempts: int = 0
    futures: int = 0


def _simulate(cfg: ExperimentConfig, layout: Layout, params: PhysicalParams, level: Level) -> Trajectory:
    backend = get_backend(cfg.physics.backend)
    return backend.simulate(
        layout,
        params,
        level,
        cfg.catalog,
        fps=cfg.render.fps,
        num_frames=cfg.render.num_frames,
        timestep=cfg.physics.timestep,
        pre_roll=cfg.physics.pre_roll,
    )


def _scale_push(layout: Layout, f: float) -> Layout:
    lay = Layout.from_dict(json.loads(to_json(layout)))
    b = lay.body("object")
    b.lin_vel = (np.asarray(b.lin_vel) * f).tolist()
    b.ang_vel = (np.asarray(b.ang_vel) * f).tolist()
    return lay


def calibrate_speed(
    cfg: ExperimentConfig,
    layout: Layout,
    params: PhysicalParams,
    level: Level,
    ep: dict[str, Any],
    iters: int = 8,
    checks_cfg: dict[str, Any] | None = None,
):
    """Simulate, and rescale the initial push until the event happens inside
    the time window. This replaces hand-tuning impulse strength per scene.

    Grows or shrinks the speed geometrically until the too-slow/too-fast
    outcomes are bracketed, then bisects (in log space) between them."""
    res = None
    traj = None
    lo, hi = None, None  # speeds known to be too slow / too fast
    for _ in range(iters):
        traj = _simulate(cfg, layout, params, level)
        res = check_physics(layout, traj, level, cfg.catalog, cfg.checks if checks_cfg is None else checks_cfg)
        if res or not res.hint:
            break
        speed = float(np.linalg.norm(layout.body("object").lin_vel))
        if res.hint == "faster":
            lo = speed
        else:
            hi = speed
        if lo is not None and hi is not None:
            new_speed = math.sqrt(lo * hi)
        else:
            new_speed = speed * (1.5 if res.hint == "faster" else 0.67)
        new_speed = float(np.clip(new_speed, ep["min_speed"], ep["max_speed"]))
        if abs(new_speed - speed) < 1e-4:
            break
        layout = _scale_push(layout, new_speed / speed)
        layout.meta["speed"] = new_speed
    return layout, traj, res


def plan_group(cfg: ExperimentConfig, event: str, scene: str, obj_key: str) -> dict[str, Any]:
    """Find a valid layout + cameras (+ futures) for one physics group."""
    level = cfg.levels[scene]
    obj = cfg.catalog.objects[obj_key]
    surface_name = choose_surface(level, event)
    surface = level.surfaces[surface_name]
    clip = cfg.render.num_frames / cfg.render.fps
    ep = event_params(event, cfg.events.get(event))
    template = EVENT_REGISTRY[event]
    checks_cfg = dict(cfg.checks)
    if cfg.benchmark.enabled:
        # The key event must happen after the shared conditioning prefix.
        from .checks import DEFAULT_CHECKS

        lo, hi = checks_cfg.get("event_window", DEFAULT_CHECKS["event_window"])
        lo = max(lo, (cfg.benchmark.prefix_frames + 3) / cfg.render.num_frames)
        checks_cfg["event_window"] = [lo, max(hi, lo + 0.1)]
    cam_cfg = cfg.raw.get("camera", {})
    last_reason = ""
    for attempt in range(cfg.design.max_layout_attempts):
        # Seed without the object so the same attempt produces similar layouts across objects.
        rng = seeded_rng(cfg.seed, "layout", event, scene, attempt)
        prop = choose_prop(cfg, level, event, seeded_rng(cfg.seed, "prop", event, scene))
        ctx = EventContext(rng, level, surface, obj, prop, clip, ep)
        try:
            layout = template(ctx)
        except ValueError as e:
            last_reason = f"template: {e}"
            continue
        params = default_params(layout, level, cfg.catalog)
        layout, traj, res = calibrate_speed(cfg, layout, params, level, ep, checks_cfg=checks_cfg)
        if not res:
            last_reason = f"physics: {res.reason}"
            continue
        if cfg.benchmark.enabled and cfg.benchmark.require_validator:
            # A reference must itself pass independent validation (this also
            # catches simulator artefacts such as energy gained at contact edges).
            from .benchmark import validate

            verdict = validate(layout, params, traj, level, cfg.catalog, cfg.benchmark.validator_tol)
            if not verdict.valid:
                last_reason = f"reference failed independent validation: {verdict.summary}"
                continue
        cams: list[CameraSpec] = []
        view_info = []
        for v in range(cfg.design.views):
            found = None
            reason = ""
            for ca in range(40):
                crng = seeded_rng(cfg.seed, "camera", event, scene, v, attempt, ca)
                cam = sample_camera(
                    crng, layout, traj, level, cfg.catalog, v, cfg.design.views, cfg.render.resolution, cam_cfg
                )
                if cam is None:
                    reason = "no camera distance fits the event"
                    continue
                vr = check_view(layout, traj, cam, level, cfg.catalog, res.info, checks_cfg)
                if vr:
                    found = cam
                    view_info.append(vr.info)
                    break
                reason = vr.reason
            if found is None:
                last_reason = f"view {v}: {reason}"
                break
            cams.append(found)
        if len(cams) < cfg.design.views:
            continue

        candidates, bench_stats = [], {}
        if cfg.benchmark.enabled:
            from .benchmark import make_candidates

            candidates, bench_stats = make_candidates(cfg, level, layout, params, traj, (event, scene, obj_key, attempt))
        futures = []
        fs = cfg.futures
        k = 0
        for fa in range(0 if cfg.benchmark.enabled else fs.count * fs.max_attempts):
            if k >= fs.count:
                break
            frng = seeded_rng(cfg.seed, "future", event, scene, obj_key, attempt, fa)
            flay, fpar, applied = perturb(layout, params, fs.perturb, frng)
            ftraj = _simulate(cfg, flay, fpar, level)
            fres = check_physics(flay, ftraj, level, cfg.catalog, checks_cfg)
            if fs.require_same_event and not fres:
                continue
            k += 1
            futures.append(
                {"index": k, "layout": flay, "params": fpar, "traj": ftraj, "applied": applied, "check": fres.info if fres else {"failed": fres.reason}}
            )
        if fs.count and k < fs.count:
            log.warning("%s/%s/%s: only %d/%d futures satisfied the event", event, scene, obj_key, k, fs.count)
        return {
            "ok": True,
            "attempt": attempt,
            "layout": layout,
            "params": params,
            "traj": traj,
            "check": res.info,
            "cameras": cams,
            "view_info": view_info,
            "futures": futures,
            "candidates": candidates,
            "bench_stats": bench_stats,
            "checks_cfg": checks_cfg,
            "prop": prop.key if prop else None,
        }
    return {"ok": False, "reason": last_reason, "attempt": cfg.design.max_layout_attempts}


# --------------------------------------------------------------------------- #
# Plan writer
# --------------------------------------------------------------------------- #
def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(obj, indent=1))


def prompt_fields(cfg: ExperimentConfig, level: Level, layout: Layout, obj: Asset, appearance: str) -> dict[str, Any]:
    s = level.surfaces[layout.surface]
    out = {
        "object_name": obj.prompt_name,
        "appearance_name": obj.appearances[appearance].prompt or appearance,
        "surface_name": s.prompt_name,
        "scene_name": level.name,
    }
    for b in layout.bodies:
        if b.role in ("collider", "occluder"):
            out[f"{b.role}_name"] = cfg.catalog.get(b.asset).prompt_name
    return out


_ENGINE_SHAPES = {
    "box": "/Engine/BasicShapes/Cube",
    "sphere": "/Engine/BasicShapes/Sphere",
    "cylinder": "/Engine/BasicShapes/Cylinder",
}


def _engine_shape(shape, center, yaw_deg: float, name: str, material) -> dict[str, Any]:
    """Unreal static mesh spec for a primitive using the 100 cm engine basic shapes."""
    he = shape.local_half_extents()
    if shape.type == "capsule":
        raise ValueError("capsule static geometry is not supported; use a cylinder")
    return {
        "name": name,
        "ue_mesh": _ENGINE_SHAPES[shape.type],
        "location": to_ue_location(center),
        "rotation": quat_to_ue_rotator(quat_from_yaw(math.radians(yaw_deg))),
        "scale": [2 * he[0], 2 * he[1], 2 * he[2]],  # metres == multiples of 100 cm
        "material": material,
    }


def static_geometry(level: Level) -> list[dict[str, Any]]:
    out = []
    for s in level.surfaces.values():
        shape, c, _ = s.box_shape()
        out.append(_engine_shape(shape, c, s.yaw_deg, f"PhysScene_surface_{s.name}", s.ue_material))
    for c in level.colliders:
        out.append(_engine_shape(c.shape, c.center, c.yaw_deg, f"PhysScene_static_{c.name}", c.ue_material))
    return out


def build_job(
    cfg: ExperimentConfig,
    group_id: str,
    layout: Layout,
    traj: Trajectory,
    traj_rel: str,
    cam: CameraSpec,
    event: str,
    scene: str,
    obj_key: str,
    appearance: str,
    view: int,
    candidate: dict[str, Any],
    check_info: dict[str, Any],
    params: PhysicalParams,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """``candidate``: {"id": "ref" | "f01" | "<violation>_<severity>", "kind": "reference" | "valid" | "invalid", ...}."""
    level = cfg.levels[scene]
    obj = cfg.catalog.objects[obj_key]
    view_name = f"v{view}"
    out_rel = f"{event}/{scene}/{obj_key}/{appearance}/{view_name}"
    cid, kind = candidate["id"], candidate["kind"]
    if kind == "valid":
        out_rel += f"/alternatives/{cid}"
    elif kind == "invalid":
        out_rel += f"/violations/{cid}"
    future = int(cid[1:]) if kind == "valid" and cid[1:].isdigit() else (0 if kind == "reference" else None)
    job_id = stable_id(cfg.name, cfg.seed, group_id, appearance, view, 0 if cid == "ref" else cid)
    bodies = []
    for b in layout.bodies:
        a = cfg.catalog.get(b.asset)
        app = a.appearances[appearance] if b.name == "object" else next(iter(a.appearances.values()))
        bodies.append(
            {
                "name": b.name,
                "asset": b.asset,
                "role": b.role,
                "stencil": b.stencil,
                "dynamic": b.dynamic,
                "ue_mesh": a.ue_mesh,
                "materials": {str(k): v for k, v in app.materials.items()},
                "scale": list(a.scale),
                "shape": a.shape.to_dict(),
                "preview_color": list(app.preview_color),
                "ue_keys": actor_keys(a, traj, b.name),
            }
        )
    r = cfg.render
    return {
        "job_id": job_id,
        "static_geometry": static_geometry(level) if level.build_geometry else [],
        "experiment": cfg.name,
        "group_id": group_id,
        "factors": {
            "event": event,
            "scene": scene,
            "object": obj_key,
            "appearance": appearance,
            "view": view_name,
            "future": future,
            "candidate": cid,
            "label": "invalid" if kind == "invalid" else "valid",
        },
        "output_rel": out_rel,
        "trajectory": traj_rel,
        "level": {"name": level.name, "ue_map": level.ue_map, "hide_actors": level.hide_actors},
        "world": world_geometry(level),
        "render": {
            "fps": r.fps,
            "num_frames": r.num_frames,
            "resolution": list(r.resolution),
            "passes": list(r.passes),
            "spatial_samples": r.spatial_samples,
            "temporal_samples": r.temporal_samples,
            "warmup_frames": r.warmup_frames,
            "motion_blur": r.motion_blur,
            "sensor_width_mm": r.sensor_width_mm,
        },
        "camera": {**asdict(cam), "intrinsics": cam.intrinsics(), "ue": camera_ue(cam)},
        "bodies": bodies,
        "metadata": {
            "event": event,
            "scene": scene,
            "object": obj_key,
            "appearance": appearance,
            "view": view_name,
            "future": future,
            "surface": layout.surface,
            **prompt_fields(cfg, level, layout, obj, appearance),
            "fps": r.fps,
            "num_frames": r.num_frames,
            "resolution": list(r.resolution),
            "mask_channels": {b.name: b.stencil - 1 for b in layout.bodies if b.stencil > 0},
            "event_frame": check_info.get("event_frame"),
            "physics": asdict(params),
            "seed": cfg.seed,
            "group_id": group_id,
            "generator": "PhysScene",
            "candidate": candidate,
            "label": "invalid" if kind == "invalid" else "valid",
            "provenance": provenance or {},
        },
    }


def _group_worker(args):
    cfg_path, event, scene, obj = args
    cfg = ExperimentConfig.load(cfg_path)
    return event, scene, obj, plan_group(cfg, event, scene, obj)


def grid(cfg: ExperimentConfig) -> list[tuple[str, str, str, str, int]]:
    cells = [
        (e, s, o, a, v)
        for e, s, o in product(cfg.design.events, cfg.design.scenes, cfg.design.objects)
        for a in cfg.appearances_for(o)
        for v in range(cfg.design.views)
    ]
    if cfg.design.sampling != "full":
        n = int(cfg.design.sampling)
        cells = random.Random(cfg.seed).sample(cells, min(n, len(cells)))
    return cells


def plan_experiment(cfg: ExperimentConfig, out_dir: str | Path, workers: int = 1) -> dict[str, Any]:
    out = Path(out_dir)
    (out / "jobs").mkdir(parents=True, exist_ok=True)
    cells = grid(cfg)
    groups = sorted({(e, s, o) for e, s, o, _, _ in cells})
    wanted = {(e, s, o): set() for e, s, o in groups}
    for e, s, o, a, v in cells:
        wanted[(e, s, o)].add((a, v))

    results: dict[tuple[str, str, str], dict[str, Any]] = {}
    if workers > 1 and cfg.source_path is not None:
        with ProcessPoolExecutor(workers) as ex:
            for e, s, o, r in ex.map(_group_worker, [(str(cfg.source_path), *g) for g in groups]):
                results[(e, s, o)] = r
    else:
        for g in groups:
            results[g] = plan_group(cfg, *g)

    manifest = []
    summary = []
    prov = provenance_info(cfg)
    for (e, s, o), res in results.items():
        gid = stable_id(cfg.name, cfg.seed, e, s, o)
        if not res["ok"]:
            log.warning("group %s/%s/%s failed after %d attempts: %s", e, s, o, res["attempt"], res["reason"])
            summary.append(asdict(GroupResult(gid, e, s, o, False, res["reason"], res["attempt"])))
            continue
        gdir = out / "groups" / gid
        _write_json(gdir / "layout.json", res["layout"])
        _write_json(gdir / "params.json", res["params"])
        _write_json(gdir / "trajectory.json", res["traj"])
        _write_json(gdir / "cameras.json", [asdict(c) for c in res["cameras"]])
        _write_json(gdir / "check.json", {"physics": res["check"], "views": res["view_info"]})
        ref_cand = {"id": "ref", "kind": "reference"}
        if cfg.benchmark.enabled:
            ref_cand["prefix_frames"] = cfg.benchmark.prefix_frames
        variants = [(ref_cand, res["layout"], res["traj"], res["check"], res["params"], f"groups/{gid}/trajectory.json")]
        for f in res["futures"]:
            fdir = gdir / "futures" / f"f{f['index']:02d}"
            _write_json(fdir / "layout.json", f["layout"])
            _write_json(fdir / "params.json", f["params"])
            _write_json(fdir / "trajectory.json", f["traj"])
            _write_json(fdir / "applied.json", f["applied"])
            cand = {"id": f"f{f['index']:02d}", "kind": "valid", "branch_frame": 0, "applied": f["applied"]}
            variants.append((cand, f["layout"], f["traj"], f["check"], f["params"], f"groups/{gid}/futures/{cand['id']}/trajectory.json"))
        for c in res["candidates"]:
            if c.kind == "reference":
                variants[0][0]["validator"] = c.applied.get("validator")
                continue
            cdir = gdir / "candidates" / c.cid
            meta = {
                "id": c.cid,
                "kind": c.kind,
                "label": c.label,
                "prefix_frames": cfg.benchmark.prefix_frames,
                "branch_frame": cfg.benchmark.prefix_frames,
                "violation": c.violation,
                "divergence_m": round(c.divergence, 5),
                "applied": {k: v for k, v in c.applied.items() if k != "validator"},
                "validator": c.applied.get("validator"),
            }
            _write_json(cdir / "layout.json", c.layout)
            _write_json(cdir / "params.json", c.params)
            _write_json(cdir / "trajectory.json", c.traj)
            _write_json(cdir / "candidate.json", meta)
            variants.append((meta, c.layout, c.traj, res["check"], c.params, f"groups/{gid}/candidates/{c.cid}/trajectory.json"))
        if res["bench_stats"]:
            _write_json(gdir / "benchmark_stats.json", res["bench_stats"])
        for a, v in sorted(wanted[(e, s, o)]):
            cam = res["cameras"][v]
            for cand, lay, traj, chk, par, trel in variants:
                job = build_job(cfg, gid, lay, traj, trel, cam, e, s, o, a, v, cand, chk, par, prov)
                _write_json(out / "jobs" / f"{job['job_id']}.json", job)
                manifest.append({"job_id": job["job_id"], "output_rel": job["output_rel"], **job["factors"]})
        g = asdict(GroupResult(gid, e, s, o, True, "", res["attempt"] + 1, len(variants) - 1))
        g["benchmark"] = res["bench_stats"]
        summary.append(g)

    with open(out / "manifest.jsonl", "w") as fh:
        for m in manifest:
            fh.write(json.dumps(m) + "\n")
    plan = {
        "experiment": cfg.name,
        "seed": cfg.seed,
        "config_path": str(cfg.source_path.resolve()) if cfg.source_path else None,
        "config": cfg.raw,
        "provenance": prov,
        "num_jobs": len(manifest),
        "groups": summary,
    }
    _write_json(out / "plan.json", plan)
    return plan
