"""PhysScene command line interface.

    physscene plan     CONFIG -o PLAN_DIR          # layouts, physics, cameras, futures -> jobs
    physscene render   PLAN_DIR --renderer preview # or --renderer unreal --uproject X.uproject
    physscene export   PLAN_DIR -o DATASET_DIR     # CRONOS-compatible dataset
    physscene validate DATASET_DIR
    physscene run      CONFIG -o OUT --renderer preview   # all of the above
    physscene inspect  PLAN_DIR                    # summary + top-down plots
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

log = logging.getLogger("physscene")


def _load_jobs(plan_dir: Path, only: list[str] | None = None) -> list[dict]:
    jobs = [json.loads(p.read_text()) for p in sorted((plan_dir / "jobs").glob("*.json"))]
    if only:
        s = set(only)
        jobs = [j for j in jobs if j["job_id"] in s]
    return jobs


def cmd_plan(a) -> int:
    from .config import ExperimentConfig
    from .planner import plan_experiment

    cfg = ExperimentConfig.load(a.config)
    t = time.time()
    plan = plan_experiment(cfg, a.output, workers=a.workers)
    ok = sum(g["ok"] for g in plan["groups"])
    print(f"planned {plan['num_jobs']} render jobs from {ok}/{len(plan['groups'])} physics groups in {time.time() - t:.1f}s -> {a.output}")
    for g in plan["groups"]:
        if not g["ok"]:
            print(f"  FAILED {g['event']}/{g['scene']}/{g['obj']}: {g['reason']}")
    return 0 if ok == len(plan["groups"]) else 2


def _preview_worker(args):
    job_path, plan_dir, render_root, scale, cfg_path = args
    from .catalog import Catalog, Level
    from .preview import render_job

    from .spec import job_fingerprint

    job = json.loads(Path(job_path).read_text())
    out = Path(render_root) / job["job_id"]
    done = out / "DONE"
    if done.exists() and done.read_text().split()[:1] == [job_fingerprint(job)]:
        return job["job_id"], "cached"
    plan = json.loads((Path(plan_dir) / "plan.json").read_text())
    base = Path(cfg_path).parent
    catalog = Catalog.load(base / plan["config"]["catalog"])
    level = next(
        lv for lv in (Level.load(base / p) for p in plan["config"]["levels"]) if lv.name == job["factors"]["scene"]
    )
    render_job(job, Path(plan_dir), out, catalog, level, scale=scale)
    return job["job_id"], "rendered"


def cmd_render(a) -> int:
    plan_dir = Path(a.plan)
    render_root = Path(a.render_root) if a.render_root else plan_dir / "renders"
    only = a.jobs.split(",") if a.jobs else None
    if a.renderer == "unreal":
        from .unreal_launcher import render_with_unreal

        if not a.uproject:
            print("--uproject is required for --renderer unreal", file=sys.stderr)
            return 1
        return render_with_unreal(plan_dir, a.uproject, a.editor, only, render_root, dry_run=a.dry_run)
    plan = json.loads((plan_dir / "plan.json").read_text())
    cfg_path = a.config or plan.get("config_path")
    if not cfg_path:
        print("cannot locate the experiment config; pass --config", file=sys.stderr)
        return 1
    paths = sorted((plan_dir / "jobs").glob("*.json"))
    if only:
        paths = [p for p in paths if p.stem in set(only)]
    if a.limit:
        paths = paths[: a.limit]
    args = [(str(p), str(plan_dir), str(render_root), a.scale, cfg_path) for p in paths]
    t = time.time()
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as ex:
            for i, (jid, st) in enumerate(ex.map(_preview_worker, args), 1):
                print(f"[{i}/{len(args)}] {jid} {st}", flush=True)
    else:
        for i, arg in enumerate(args, 1):
            jid, st = _preview_worker(arg)
            print(f"[{i}/{len(args)}] {jid} {st}", flush=True)
    print(f"rendered {len(args)} jobs in {time.time() - t:.1f}s -> {render_root}")
    return 0


def cmd_export(a) -> int:
    from .export import export_plan

    out = export_plan(a.plan, a.output, a.render_root, keep_frames=not a.no_frames)
    print(f"exported {len(out)} samples -> {a.output}")
    return 0


def cmd_validate(a) -> int:
    from .validate import validate_dataset

    rep = validate_dataset(a.dataset)
    print(rep.summary())
    return 0 if rep.ok else 1


def cmd_inspect(a) -> int:
    from .inspect import inspect_plan

    inspect_plan(Path(a.plan), Path(a.output) if a.output else None)
    return 0


def cmd_run(a) -> int:
    out = Path(a.output)
    plan_dir, dataset = out / "plan", out / "dataset"
    ns = argparse.Namespace
    rc = cmd_plan(ns(config=a.config, output=plan_dir, workers=a.workers))
    if rc == 1:
        return rc
    rc = cmd_render(
        ns(plan=plan_dir, renderer=a.renderer, render_root=None, jobs=None, uproject=a.uproject, editor=a.editor,
           dry_run=False, config=a.config, scale=a.scale, workers=a.workers, limit=a.limit)
    )
    if rc:
        return rc
    cmd_export(ns(plan=plan_dir, output=dataset, render_root=None, no_frames=False))
    return cmd_validate(ns(dataset=dataset))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="physscene", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("plan", help="generate layouts, physics, cameras and render jobs")
    sp.add_argument("config")
    sp.add_argument("-o", "--output", required=True)
    sp.add_argument("--workers", type=int, default=1)
    sp.set_defaults(fn=cmd_plan)

    def render_args(sp, with_plan=True):
        sp.add_argument("--renderer", choices=["preview", "unreal"], default="preview")
        sp.add_argument("--uproject", help="Unreal project (.uproject) for --renderer unreal")
        sp.add_argument("--editor", help="path to UnrealEditor executable (or set UE_EDITOR)")
        sp.add_argument("--scale", type=float, default=0.25, help="preview renderer resolution scale")
        sp.add_argument("--workers", type=int, default=1)
        sp.add_argument("--limit", type=int, default=0, help="render only the first N jobs")

    sp = sub.add_parser("render", help="render planned jobs")
    sp.add_argument("plan")
    render_args(sp)
    sp.add_argument("--render-root")
    sp.add_argument("--jobs", help="comma-separated job ids")
    sp.add_argument("--config", help="experiment config (defaults to the one recorded in plan.json)")
    sp.add_argument("--dry-run", action="store_true", help="print the Unreal command without running it")
    sp.set_defaults(fn=cmd_render)

    sp = sub.add_parser("export", help="convert renders into the dataset layout")
    sp.add_argument("plan")
    sp.add_argument("-o", "--output", required=True)
    sp.add_argument("--render-root")
    sp.add_argument("--no-frames", action="store_true", help="skip per-frame RGB PNGs (keep mp4 only)")
    sp.set_defaults(fn=cmd_export)

    sp = sub.add_parser("validate", help="check a dataset for completeness / CRONOS compatibility")
    sp.add_argument("dataset")
    sp.set_defaults(fn=cmd_validate)

    sp = sub.add_parser("inspect", help="summarise a plan and draw top-down layout plots")
    sp.add_argument("plan")
    sp.add_argument("-o", "--output", help="directory for plots (default: PLAN/inspect)")
    sp.set_defaults(fn=cmd_inspect)

    sp = sub.add_parser("run", help="plan + render + export + validate")
    sp.add_argument("config")
    sp.add_argument("-o", "--output", required=True)
    render_args(sp)
    sp.set_defaults(fn=cmd_run)

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
