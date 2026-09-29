"""Movie Render Queue batch driver.

* jobs come from a PhysScene plan (``<plan>/jobs/*.json``);
* jobs are grouped by level: load the level, build sequences, queue, render;
* each job is rendered as up to three MRQ jobs: ``rgb`` (final image,
  anti-aliased), ``labels`` (stencil-bit masks, no AA) and ``depth`` (EXR);
* finished jobs get a ``DONE`` marker, so an interrupted batch resumes where it
  stopped;
* when everything is done, the editor quits (if ``PHYSSCENE_QUIT_WHEN_DONE=1``).
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict, defaultdict
from pathlib import Path

import unreal

from .materials import depth_material, stencil_material
from .sequence import build_sequence
from .ue_compat import actor_subsystem, level_subsystem, log, set_prop, warn

KINDS_FOR_PASS = {"rgb": "rgb", "mask": "labels", "depth": "depth"}


def job_fingerprint(job: dict) -> str:
    """Must match physscene.spec.job_fingerprint."""
    return hashlib.sha1(json.dumps(job, sort_keys=True).encode()).hexdigest()[:16]


def _is_done(render_root: str, job: dict) -> bool:
    p = Path(render_root, job["job_id"], "DONE")
    return p.exists() and p.read_text().split()[:1] == [job_fingerprint(job)]


def load_jobs(plan_dir: str, render_root: str, only: set[str] | None = None, force: bool = False) -> list[dict]:
    jobs = []
    for p in sorted(Path(plan_dir, "jobs").glob("*.json")):
        job = json.loads(p.read_text())
        if only and job["job_id"] not in only:
            continue
        if not force and _is_done(render_root, job):
            continue
        jobs.append(job)
    return jobs


def _console_vars(config, cvars: dict) -> None:
    s = config.find_or_add_setting_by_class(unreal.MoviePipelineConsoleVariableSetting)
    for k, v in cvars.items():
        if hasattr(s, "add_or_update_console_variable"):
            s.add_or_update_console_variable(k, float(v))
        else:  # older API: map property
            m = dict(s.get_editor_property("console_variables"))
            m[k] = float(v)
            s.set_editor_property("console_variables", m)


def configure(mrq_job, job: dict, kind: str, out_dir: str) -> None:
    r = job["render"]
    w, h = r["resolution"]
    config = mrq_job.get_configuration()

    out = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
    out.set_editor_property("output_directory", unreal.DirectoryPath(out_dir))
    out.set_editor_property("file_name_format", "{render_pass}/{frame_number}")
    out.set_editor_property("output_resolution", unreal.IntPoint(w, h))
    set_prop(out, "zero_pad_frame_numbers", 4)
    set_prop(out, "use_custom_frame_rate", True)
    set_prop(out, "output_frame_rate", unreal.FrameRate(int(r["fps"]), 1))
    set_prop(out, "override_existing_output", True)
    set_prop(out, "use_custom_playback_range", True)
    set_prop(out, "custom_start_frame", 0)
    set_prop(out, "custom_end_frame", int(r["num_frames"]))

    deferred = config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
    aa = config.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
    set_prop(aa, "engine_warm_up_count", int(r.get("warmup_frames", 32)))
    set_prop(aa, "render_warm_up_count", int(r.get("warmup_frames", 32)))
    set_prop(aa, "use_camera_cut_for_warm_up", False)

    if kind == "rgb":
        config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)
        set_prop(aa, "spatial_sample_count", int(r.get("spatial_samples", 1)))
        set_prop(aa, "temporal_sample_count", int(r.get("temporal_samples", 8)))
    else:
        mat = stencil_material() if kind == "labels" else depth_material()
        pp = unreal.MoviePipelinePostProcessPass()
        pp.set_editor_property("enabled", True)
        pp.set_editor_property("material", mat)
        deferred.set_editor_property("additional_post_process_materials", [pp])
        set_prop(deferred, "render_main_pass", False)
        set_prop(deferred, "disable_multisample_effects", True)
        set_prop(aa, "spatial_sample_count", 1)
        set_prop(aa, "temporal_sample_count", 1)
        set_prop(aa, "override_anti_aliasing", True)
        set_prop(aa, "anti_aliasing_method", unreal.AntiAliasingMethod.AAM_NONE)
        if kind == "labels":
            config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)
        else:
            config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_EXR)

    cvars = {"r.CustomDepth": 3}
    if not r.get("motion_blur"):
        cvars["r.MotionBlurQuality"] = 0
    _console_vars(config, cvars)
    config.find_or_add_setting_by_class(unreal.MoviePipelineGameOverrideSetting)


class BatchRenderer:
    """State machine: one level at a time, all of its jobs in one MRQ queue."""

    def __init__(self, plan_dir: str, render_root: str, jobs: list[dict], quit_when_done: bool = False, use_spawnables: bool = True):
        self.plan_dir = plan_dir
        self.render_root = render_root
        self.quit_when_done = quit_when_done
        self.use_spawnables = use_spawnables
        by_level: dict[str, list[dict]] = OrderedDict()
        for j in jobs:
            by_level.setdefault(j["level"]["ue_map"], []).append(j)
        self.batches = list(by_level.items())
        self.pending: dict[str, set[str]] = defaultdict(set)
        self.leftovers = []
        self.executor = None
        self.subsystem = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
        self.failed: list[str] = []
        self.fingerprints = {j["job_id"]: job_fingerprint(j) for j in jobs}

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        log(f"{sum(len(b) for _, b in self.batches)} jobs in {len(self.batches)} level batches")
        self._next_batch()

    def _next_batch(self) -> None:
        self._cleanup_leftovers()
        if not self.batches:
            self._finish()
            return
        ue_map, jobs = self.batches.pop(0)
        log(f"loading level {ue_map} ({len(jobs)} jobs)")
        level_subsystem().load_level(ue_map)
        hide = set(jobs[0]["level"].get("hide_actors", []))
        if hide:
            for a in actor_subsystem().get_all_level_actors():
                if a.get_actor_label() in hide:
                    a.set_actor_hidden_in_game(True)

        queue = self.subsystem.get_queue()
        queue.delete_all_jobs()
        for job in jobs:
            try:
                seq, left = build_sequence(job, self.use_spawnables)
            except Exception as e:  # noqa: BLE001
                warn(f"job {job['job_id']}: failed to build sequence: {e}")
                self.failed.append(job["job_id"])
                continue
            self.leftovers += left
            kinds = ["rgb"] + [KINDS_FOR_PASS[p] for p in job["render"]["passes"] if p in ("mask", "depth")]
            for kind in kinds:
                mj = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
                mj.set_editor_property("map", unreal.SoftObjectPath(ue_map))
                mj.set_editor_property("sequence", unreal.SoftObjectPath(seq.get_path_name()))
                mj.set_editor_property("job_name", f"{job['job_id']}_{kind}")
                set_prop(mj, "user_data", json.dumps({"job_id": job["job_id"], "kind": kind}))
                configure(mj, job, kind, os.path.join(self.render_root, job["job_id"], kind))
                self.pending[job["job_id"]].add(kind)

        self.executor = unreal.MoviePipelinePIEExecutor(self.subsystem)
        self.executor.on_executor_finished_delegate.add_callable_unique(self._on_batch_finished)
        self.executor.on_individual_job_work_finished_delegate.add_callable_unique(self._on_job_finished)
        self.subsystem.render_queue_with_executor_instance(self.executor)

    def _on_job_finished(self, output_data) -> None:
        try:
            mj = output_data.get_editor_property("job")
            info = json.loads(mj.get_editor_property("user_data") or "{}")
            success = output_data.get_editor_property("success")
        except Exception as e:  # noqa: BLE001
            warn(f"could not read job output data: {e}")
            return
        jid, kind = info.get("job_id"), info.get("kind")
        if not jid:
            return
        if not success:
            warn(f"job {jid} pass {kind} failed")
            self.failed.append(jid)
            return
        self.pending[jid].discard(kind)
        if not self.pending[jid]:
            d = Path(self.render_root, jid)
            d.mkdir(parents=True, exist_ok=True)
            (d / "DONE").write_text(self.fingerprints.get(jid, "?") + " unreal\n")
            log(f"job {jid} done")

    def _on_batch_finished(self, executor, success) -> None:
        log(f"level batch finished (success={success})")
        self._next_batch()

    def _cleanup_leftovers(self) -> None:
        for a in self.leftovers:
            try:
                actor_subsystem().destroy_actor(a)
            except Exception:  # noqa: BLE001
                pass
        self.leftovers = []

    def _finish(self) -> None:
        summary = {"failed": sorted(set(self.failed))}
        Path(self.render_root).mkdir(parents=True, exist_ok=True)
        Path(self.render_root, "unreal_summary.json").write_text(json.dumps(summary, indent=1))
        log(f"all batches finished; {len(summary['failed'])} failed jobs")
        if self.quit_when_done:
            unreal.SystemLibrary.quit_editor()


_ACTIVE = None  # keep a reference so delegates are not garbage collected


def run_from_env() -> None:
    global _ACTIVE
    plan = os.environ.get("PHYSSCENE_PLAN")
    if not plan:
        raise RuntimeError("PHYSSCENE_PLAN is not set")
    root = os.environ.get("PHYSSCENE_RENDER_ROOT", os.path.join(plan, "renders"))
    only = set(filter(None, os.environ.get("PHYSSCENE_JOBS", "").split(","))) or None
    force = os.environ.get("PHYSSCENE_FORCE") == "1"
    jobs = load_jobs(plan, root, only, force)
    quit_when_done = os.environ.get("PHYSSCENE_QUIT_WHEN_DONE") == "1"
    spawnables = os.environ.get("PHYSSCENE_POSSESSABLES") != "1"
    if not jobs:
        log("nothing to render (all jobs done?)")
        if quit_when_done:
            unreal.SystemLibrary.quit_editor()
        return
    _ACTIVE = BatchRenderer(plan, root, jobs, quit_when_done, spawnables)
    _ACTIVE.start()
