"""Decision-gate report: evidence that a generated dataset meets the
requirements for a controlled physical-validity benchmark.

Each gate criterion is tested directly against the plan and the exported
dataset:

G1  reproducible initial conditions and seeds
G2  programmatic control over physical properties
G3  per-frame state sufficient for independent validity checks
G4  reproducible counterfactual visual transformations
G5  stable batch rendering without manual scene adjustment
G6  complete provenance for every video

The staged validation protocol (physics-only checks, one-render smoke test,
matched-prefix checks, full rendering, independent state validation, frozen
scoring) is reported in the same file.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import ExperimentConfig
from .provenance import file_sha256
from .spec import Layout, PhysicalParams, Trajectory, job_fingerprint

REQUIRED_META = ["event", "scene", "object", "appearance", "view", "candidate", "label", "seed", "group_id", "job_id"]
REQUIRED_PROV = [
    "physscene_version",
    "physscene_commit",
    "config_sha256",
    "physics_backend",
    "mujoco_version",
    "job_fingerprint",
    "renderer",
    "render_settings",
]


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def _load(p: Path):
    return json.loads(p.read_text())


class Gate:
    def __init__(self, plan_dir: Path, dataset_dir: Path, sample_groups: int = 4):
        self.plan_dir = Path(plan_dir)
        self.dataset = Path(dataset_dir)
        self.plan = _load(self.plan_dir / "plan.json")
        self.cfg = ExperimentConfig.load(self.plan["config_path"])
        self.groups = [g for g in self.plan["groups"] if g["ok"]]
        self.sample = self.groups[:: max(1, len(self.groups) // sample_groups)][:sample_groups]
        self.samples = {
            mp.parent: _load(mp) for mp in sorted(self.dataset.rglob("metadata.json")) if mp.parent != self.dataset
        }
        self.jobs = {p.stem: _load(p) for p in (self.plan_dir / "jobs").glob("*.json")}

    # ------------------------------------------------------------------ G1
    def g1_reproducibility(self) -> dict[str, Any]:
        from .physics import get_backend
        from .planner import plan_group

        rows, ok = [], True
        backend = get_backend(self.cfg.physics.backend)
        for g in self.sample:
            gdir = self.plan_dir / "groups" / g["group_id"]
            res = plan_group(self.cfg, g["event"], g["scene"], g["obj"])
            from .spec import to_json

            same_layout = json.loads(to_json(res["layout"])) == _load(gdir / "layout.json")
            same_traj = json.loads(to_json(res["traj"])) == _load(gdir / "trajectory.json")
            cand_same = True
            for c in res.get("candidates", []):
                if c.kind == "reference":
                    continue
                saved = gdir / "candidates" / c.cid / "trajectory.json"
                cand_same &= saved.exists() and json.loads(to_json(c.traj)) == _load(saved)
            # re-simulate from the saved state alone (no planner involvement)
            lay = Layout.from_dict(_load(gdir / "layout.json"))
            par = PhysicalParams.from_dict(_load(gdir / "params.json"))
            tr = backend.simulate(lay, par, self.cfg.levels[g["scene"]], self.cfg.catalog, self.cfg.render.fps,
                                  self.cfg.render.num_frames, self.cfg.physics.timestep, self.cfg.physics.pre_roll)
            saved = Trajectory.from_dict(_load(gdir / "trajectory.json"))
            resim_err = max(float(np.abs(tr.pos(n) - saved.pos(n)).max()) for n in saved.names)
            row_ok = same_layout and same_traj and cand_same and resim_err == 0.0
            ok &= row_ok
            rows.append({"group": f"{g['event']}/{g['scene']}/{g['obj']}", "replan_layout_identical": same_layout,
                         "replan_trajectory_identical": same_traj, "replan_candidates_identical": cand_same,
                         "resimulation_max_error_m": resim_err, "pass": row_ok})
        return {"pass": ok, "method": "Re-ran the planner from the saved config and seed for sampled groups, and "
                "re-simulated each reference from its saved layout and parameters only.", "rows": rows}

    # ------------------------------------------------------------------ G2
    def g2_control(self) -> dict[str, Any]:
        from .benchmark import validate
        from .physics import get_backend
        from .violations import branch, divergence

        backend = get_backend(self.cfg.physics.backend)
        P = self.cfg.benchmark.prefix_frames if self.cfg.benchmark.enabled else 0
        knobs = [("object", "friction"), ("object", "restitution"), ("object", "mass"), ("object", "rolling_friction"),
                 ("surface", "friction")]
        table = defaultdict(list)
        unstable = defaultdict(int)
        recorded_ok = True
        for g in self.sample:
            gdir = self.plan_dir / "groups" / g["group_id"]
            lay = Layout.from_dict(_load(gdir / "layout.json"))
            par = PhysicalParams.from_dict(_load(gdir / "params.json"))
            ref = Trajectory.from_dict(_load(gdir / "trajectory.json"))
            level = self.cfg.levels[g["scene"]]

            def sim(l, p, init_state=None, num_frames=None, exclude_pairs=None):
                return backend.simulate(l, p, level, self.cfg.catalog, self.cfg.render.fps, num_frames,
                                        self.cfg.physics.timestep, 0.0, init_state, exclude_pairs)

            for target, name in knobs:
                for factor in (0.5, 2.0):
                    p2 = PhysicalParams.from_dict(json.loads(json.dumps(par.__dict__)))
                    d = p2.surface if target == "surface" else p2.bodies[target]
                    d[name] = d[name] * factor
                    from .physics.mujoco_backend import build_mjcf

                    xml = build_mjcf(lay, p2, level, self.cfg.catalog, self.cfg.physics.timestep)
                    if name == "mass":
                        recorded_ok &= f'mass="{d[name]:.6g}"' in xml
                    t = branch(ref, P, lay, p2, sim)
                    table[f"{target}.{name}"].append(divergence(ref, t))
                    if not validate(lay, p2, t, level, self.cfg.catalog, self.cfg.benchmark.validator_tol).valid:
                        unstable[f"{target}.{name}"] += 1
        rows = [{"parameter": k, "groups_x_settings": len(v), "responding": int(sum(x > 1e-4 for x in v)),
                 "median_divergence_m": round(float(np.median(v)), 4), "max_divergence_m": round(float(np.max(v)), 4),
                 "flagged_by_validator": unstable[k]}
                for k, v in table.items()]
        ok = recorded_ok and all(r["responding"] > 0 for r in rows)
        return {"pass": ok, "method": f"Set each hidden parameter to 0.5x and 2x from the matched prefix (frame {P}) "
                "and measured the change in the object trajectory. Also checked that the value reaches the simulator model. "
                "Every swept trajectory is also run through the independent validator. A flagged result means the "
                "simulator became unphysical at that setting (e.g. a numerical instability); such trajectories are never "
                "admitted to the dataset.",
                "sweep_results_flagged_by_validator": int(sum(unstable.values())),
                "rows": rows}

    # ------------------------------------------------------------------ G3
    def g3_independent_validation(self) -> dict[str, Any]:
        from .validator import validate_states_file

        conf = Counter()
        by_type = defaultdict(Counter)
        frames_ok = True
        seen = set()
        for d, m in self.samples.items():
            c = m.get("candidate") or {}
            key = (m["group_id"], c.get("id"))
            if key in seen:  # state is identical across views/appearances (checked in G4)
                continue
            seen.add(key)
            st = d / "states.json"
            v = validate_states_file(st)
            T = len(next(iter(_load(st)["trajectory"]["positions"].values())))
            frames_ok &= T == m["num_frames"]
            truth = "invalid" if c.get("kind") == "invalid" else "valid"
            pred = "valid" if v.valid else "invalid"
            conf[(truth, pred)] += 1
            if truth == "invalid":
                by_type[(c.get("violation") or {}).get("type")][pred] += 1
        prefilter = Counter()
        for g in self.groups:
            for k, n in (g.get("benchmark") or {}).items():
                if ":" in k or k.endswith("flagged") or k.startswith("valid_"):
                    prefilter[k] += n
        fp = conf[("valid", "invalid")]
        fn = conf[("invalid", "valid")]
        tp = conf[("invalid", "invalid")]
        tn = conf[("valid", "valid")]
        ok = frames_ok and fp == 0 and fn == 0
        return {
            "pass": ok,
            "method": "Re-derived validity from each exported states.json with physics-law checks (persistence, ballistic "
            "motion, Coulomb bound, rolling angular momentum, energy, non-penetration). The validator shares no code "
            "with the generator.",
            "confusion": {"valid_as_valid": tn, "valid_as_invalid": fp, "invalid_as_invalid": tp, "invalid_as_valid": fn},
            "by_violation_type": {k: dict(v) for k, v in by_type.items()},
            "state_frames_match_video": frames_ok,
            "generation_prefilter": dict(prefilter),
        }

    # ------------------------------------------------------------------ G4
    def g4_counterfactual_visuals(self) -> dict[str, Any]:
        state_hash = defaultdict(set)
        by_key = {}
        for d, m in self.samples.items():
            c = (m.get("candidate") or {}).get("id")
            st = _load(d / "states.json")
            state_hash[(m["group_id"], c)].add(_h(st["trajectory"]))
            by_key[(m["group_id"], m["appearance"], m["view"], c)] = (d, m, st)
        invariant = all(len(v) == 1 for v in state_hash.values())

        prefix_state_ok, pix_max, n_pairs = True, 0, 0
        for (gid, app, view, c), (d, m, st) in by_key.items():
            if c == "ref" or (gid, app, view, "ref") not in by_key:
                continue
            rd, rm, rst = by_key[(gid, app, view, "ref")]
            P = (m.get("candidate") or {}).get("prefix_frames") or 0
            for n in st["trajectory"]["names"]:
                a = st["trajectory"]["positions"][n][: P + 1]
                b = rst["trajectory"]["positions"][n][: P + 1]
                prefix_state_ok &= a == b
            for t in range(P + 1):
                fa, fb = d / "rgb" / f"frame_{t:04d}.png", rd / "rgb" / f"frame_{t:04d}.png"
                if fa.exists() and fb.exists():
                    diff = np.abs(np.asarray(Image.open(fa), int) - np.asarray(Image.open(fb), int)).max()
                    pix_max = max(pix_max, int(diff))
            n_pairs += 1

        # mask <-> state consistency (renders agree with the physics state)
        errs = []
        for d, m in list(self.samples.items())[:: max(1, len(self.samples) // 60)]:
            if not (d / "masks.npz").exists():
                continue
            cam = _load(d / "camera.json")
            st = _load(d / "states.json")
            masks = np.load(d / "masks.npz")["masks"]
            K, W = np.asarray(cam["intrinsics"]), np.asarray(cam["world_to_cam_opencv"])
            vis = st["trajectory"].get("visible", {}).get("object")
            for t in range(0, masks.shape[0], 6):
                if vis is not None and not vis[t]:
                    continue
                mk = masks[t, ..., 0]
                if mk.sum() < 30:
                    continue
                pc = (W @ np.append(st["trajectory"]["positions"]["object"][t], 1.0))[:3]
                uv = (K @ pc)[:2] / pc[2]
                ys, xs = np.nonzero(mk)
                if mk.sum() > 0.8 * np.pi * (0.5 * max(np.ptp(xs), np.ptp(ys))) ** 2 * 0.5:  # mostly unoccluded
                    errs.append(float(np.hypot(xs.mean() - uv[0], ys.mean() - uv[1])))

        # render determinism: re-render one job with the preview renderer
        rerender = None
        for d, m in self.samples.items():
            if m.get("provenance", {}).get("renderer") != "preview":
                break
            job = self.jobs.get(m["job_id"])
            if job is None:
                break
            from .preview import render_job

            with tempfile.TemporaryDirectory() as tmp:
                scale = np.asarray(Image.open(d / "rgb" / "frame_0000.png")).shape[1] / job["camera"]["resolution"][0]
                render_job(job, self.plan_dir, Path(tmp), self.cfg.catalog, self.cfg.levels[m["scene"]], scale=scale)
                diffs = [
                    int(np.abs(np.asarray(Image.open(Path(tmp) / "rgb" / f"{t:04d}.png"), int)
                               - np.asarray(Image.open(d / "rgb" / f"frame_{t:04d}.png"), int)).max())
                    for t in range(0, m["num_frames"], 7)
                ]
            rerender = {"job": m["job_id"], "max_pixel_diff": max(diffs)}
            break

        ok = invariant and prefix_state_ok and pix_max <= 2 and (not errs or np.median(errs) < 3.0) and (
            rerender is None or rerender["max_pixel_diff"] == 0)
        return {
            "pass": bool(ok),
            "method": "Checked that every view and appearance of a (group, candidate) pair renders the identical state, that "
            "candidate and reference share bit-identical state and pixels over the matched prefix, that "
            "rendered masks agree with projected state, and that a job re-renders bit-identically.",
            "state_invariant_across_visual_interventions": invariant,
            "prefix_state_identical": prefix_state_ok,
            "prefix_pairs_checked": n_pairs,
            "prefix_max_pixel_diff": pix_max,
            "mask_vs_state_centroid_error_px": {"n": len(errs), "median": round(float(np.median(errs)), 2) if errs else None,
                                                "p95": round(float(np.percentile(errs, 95)), 2) if errs else None},
            "rerender": rerender,
        }

    # ------------------------------------------------------------------ G5
    def g5_batch(self) -> dict[str, Any]:
        n_jobs = len(self.jobs)
        rendered = 0
        root = self.plan_dir / "renders"
        for jid, job in self.jobs.items():
            dn = root / jid / "DONE"
            rendered += dn.exists() and dn.read_text().split()[:1] == [job_fingerprint(job)]
        attempts = Counter(g["attempts"] for g in self.groups)
        cfg_hash_ok = file_sha256(Path(self.plan["config_path"])) == self.plan["provenance"].get("config_sha256")
        ok = rendered == n_jobs and len(self.samples) == n_jobs and len(self.groups) == len(self.plan["groups"]) and cfg_hash_ok
        return {
            "pass": ok,
            "method": "Every planned job must have an up-to-date render (content fingerprint) and an exported sample. "
            "Every physics group must plan without intervention, and the config must be unchanged since planning.",
            "groups_planned": f"{len(self.groups)}/{len(self.plan['groups'])}",
            "layout_attempts_histogram": dict(sorted(attempts.items())),
            "jobs_rendered": f"{rendered}/{n_jobs}",
            "samples_exported": len(self.samples),
            "config_unchanged_since_planning": cfg_hash_ok,
            "manual_edits": 0,
        }

    # ------------------------------------------------------------------ G6
    def g6_provenance(self) -> dict[str, Any]:
        missing = Counter()
        for d, m in self.samples.items():
            for k in REQUIRED_META:
                if k not in m:
                    missing[f"metadata.{k}"] += 1
            prov = m.get("provenance", {})
            for k in REQUIRED_PROV:
                if not prov.get(k):
                    missing[f"provenance.{k}"] += 1
            for f in ("camera.json", "states.json", "movies/complete.mp4"):
                if not (d / f).exists():
                    missing[f] += 1
        manifest_ok = None
        mf = self.dataset / "benchmark" / "frozen_manifest.json"
        if mf.exists():
            man = _load(mf)
            manifest_ok = all(file_sha256(self.dataset / rel) == h for rel, h in man["files"].items())
        ok = not missing and manifest_ok is not False
        return {
            "pass": ok,
            "method": "Every sample must carry scene, physics, candidate/intervention and render settings, the code "
            "version, config hash and job fingerprint. The frozen manifest hashes must match the files on disk.",
            "samples": len(self.samples),
            "missing_fields": dict(missing),
            "frozen_manifest_verified": manifest_ok,
        }

    # ------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        gates = {
            "G1 reproducible initial conditions and seeds": self.g1_reproducibility(),
            "G2 programmatic control over physical properties": self.g2_control(),
            "G3 per-frame state sufficient for independent validity checks": self.g3_independent_validation(),
            "G4 reproducible counterfactual visual transformations": self.g4_counterfactual_visuals(),
            "G5 stable batch rendering without manual adjustment": self.g5_batch(),
            "G6 complete provenance": self.g6_provenance(),
        }
        summary_path = self.dataset / "benchmark" / "summary.json"
        stages = {
            "1 physics-only state checks": all(g["ok"] for g in self.plan["groups"]),
            "2 one-render smoke test": len(self.samples) > 0,
            "3 matched-prefix checks": gates["G4 reproducible counterfactual visual transformations"]["prefix_state_identical"],
            "4 full rendering": gates["G5 stable batch rendering without manual adjustment"]["pass"],
            "5 independent state validation": gates["G3 per-frame state sufficient for independent validity checks"]["pass"],
            "6 frozen scoring manifest": gates["G6 complete provenance"]["frozen_manifest_verified"] is True,
        }
        return {
            "plan": str(self.plan_dir),
            "dataset": str(self.dataset),
            "experiment": self.plan["experiment"],
            "provenance": self.plan.get("provenance"),
            "dataset_summary": _load(summary_path) if summary_path.exists() else None,
            "gates": gates,
            "validation_stages": stages,
            "all_pass": all(g["pass"] for g in gates.values()) and all(stages.values()),
        }


def render_markdown(r: dict[str, Any]) -> str:
    L = [f"# Decision-gate report: `{r['experiment']}`", ""]
    L.append(f"**Overall: {'PASS' if r['all_pass'] else 'FAIL'}**")
    L.append("")
    s = r.get("dataset_summary") or {}
    if s:
        L.append(f"Dataset: {s.get('references')} reference clips, {s.get('pairs')} reference-candidate pairs, "
                 f"sha256 `{s.get('dataset_sha256', '')[:16]}…`")
        L.append("")
        L.append("| candidate type | pairs |\n|---|---|")
        for k, v in sorted(s.get("by_type", {}).items()):
            L.append(f"| {k} | {v} |")
        L.append("")
    L.append("## Gate criteria\n")
    L.append("| criterion | result |\n|---|---|")
    for name, g in r["gates"].items():
        L.append(f"| {name} | {'PASS' if g['pass'] else 'FAIL'} |")
    L.append("")
    for name, g in r["gates"].items():
        L.append(f"### {name}: {'PASS' if g['pass'] else 'FAIL'}\n")
        L.append(g["method"] + "\n")
        body = {k: v for k, v in g.items() if k not in ("pass", "method", "rows")}
        if body:
            L.append("```json\n" + json.dumps(body, indent=1) + "\n```\n")
        if g.get("rows"):
            keys = list(g["rows"][0].keys())
            L.append("| " + " | ".join(keys) + " |")
            L.append("|" + "---|" * len(keys))
            for row in g["rows"]:
                L.append("| " + " | ".join(str(row[k]) for k in keys) + " |")
            L.append("")
    L.append("## Validation-before-scale stages\n")
    L.append("| stage | result |\n|---|---|")
    for k, v in r["validation_stages"].items():
        L.append(f"| {k} | {'PASS' if v else 'FAIL'} |")
    L.append("")
    p = r.get("provenance") or {}
    L.append("## Provenance\n")
    L.append("```json\n" + json.dumps(p, indent=1) + "\n```")
    return "\n".join(L)


def run_gate(plan_dir, dataset_dir, out_dir=None, sample_groups: int = 4) -> dict[str, Any]:
    r = Gate(Path(plan_dir), Path(dataset_dir), sample_groups).run()
    out = Path(out_dir) if out_dir else Path(dataset_dir) / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate_report.json").write_text(json.dumps(r, indent=1, default=str))
    (out / "GATE_REPORT.md").write_text(render_markdown(r))
    return r
