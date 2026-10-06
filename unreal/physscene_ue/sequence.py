"""Build one Level Sequence per job: bodies, optional procedural geometry and
the camera, all as spawnables with baked transform keys.

Baked keys (rather than live Chaos simulation during each render) guarantee that every viewpoint and
appearance of an event shows exactly the same motion, and that renders are
bit-for-bit repeatable. Spawnables live inside the sequence asset, so the
level is never modified or saved.
"""
from __future__ import annotations

import math

import unreal

from .materials import GENERATED, load_material
from .ue_compat import (
    actor_subsystem,
    add_key,
    add_sequence_track,
    add_spawnable,
    asset_tools,
    binding_id,
    double_channels,
    log,
    set_prop,
    set_section_range,
    warn,
)


def _spawn_mesh_actor(name: str, mesh_path: str, materials: dict, scale, stencil: int, location, rotation):
    sub = actor_subsystem()
    actor = sub.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(*location), unreal.Rotator(*rotation))
    actor.set_actor_label(name)
    smc = actor.static_mesh_component
    smc.set_mobility(unreal.ComponentMobility.MOVABLE)
    mesh = unreal.load_asset(mesh_path)
    if mesh is None:
        warn(f"mesh not found: {mesh_path}")
    else:
        smc.set_static_mesh(mesh)
    for slot, spec in (materials or {}).items():
        mat = load_material(spec)
        if mat is not None:
            smc.set_material(int(slot), mat)
    smc.set_simulate_physics(False)
    smc.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
    smc.set_render_custom_depth(stencil > 0)
    smc.set_custom_depth_stencil_value(int(stencil))
    actor.set_actor_scale3d(unreal.Vector(*scale))
    actor.tags = [unreal.Name("PhysScene")]
    return actor


def _bind(sequence, actor, use_spawnables: bool):
    binding = add_spawnable(sequence, actor) if use_spawnables else None
    if binding is not None:
        actor_subsystem().destroy_actor(actor)
        return binding, None
    return sequence.add_possessable(actor), actor


def _transform_keys(binding, n_frames: int, locations, rotations, scale, visible=None) -> None:
    """Bake per-frame keys. Around frames where ``visible`` changes, keys are
    stepped (constant interpolation), so sub-frame samples never sweep a
    hidden body between its real position and its parking spot."""
    track = binding.add_track(unreal.MovieScene3DTransformTrack)
    section = track.add_section()
    set_section_range(section, 0, n_frames)
    ch = double_channels(section)
    if len(ch) < 9:
        raise RuntimeError(f"unexpected transform channel count {len(ch)}")
    n = len(locations)
    for f, (loc, rot) in enumerate(zip(locations, rotations)):
        step = bool(visible) and (
            (f + 1 < n and visible[f] != visible[f + 1]) or not visible[f]
        )
        for i in range(3):
            add_key(ch[i], f, loc[i], constant=step)
            add_key(ch[3 + i], f, rot[i], constant=step)
    for i in range(3):
        add_key(ch[6 + i], 0, scale[i])


def _camera(job, sequence, n_frames: int, use_spawnables: bool):
    cam = job["camera"]["ue"]
    w, h = job["render"]["resolution"]
    sensor_w = float(job["render"].get("sensor_width_mm", 36.0))
    sub = actor_subsystem()
    actor = sub.spawn_actor_from_class(unreal.CineCameraActor, unreal.Vector(*cam["location"]), unreal.Rotator(*cam["rotation"]))
    actor.set_actor_label("PhysScene_Camera")
    comp = actor.get_cine_camera_component()
    fb = comp.get_editor_property("filmback")
    fb.set_editor_property("sensor_width", sensor_w)
    fb.set_editor_property("sensor_height", sensor_w * h / w)
    comp.set_editor_property("filmback", fb)
    focal = (sensor_w / 2.0) / math.tan(math.radians(cam["hfov_deg"]) / 2.0)
    comp.set_editor_property("current_focal_length", focal)
    fs = comp.get_editor_property("focus_settings")
    fs.set_editor_property("focus_method", unreal.CameraFocusMethod.DISABLE)
    comp.set_editor_property("focus_settings", fs)
    pps = comp.get_editor_property("post_process_settings")
    # MRQ uses the camera's motion blur amount as the shutter fraction.
    set_prop(pps, "override_motion_blur_amount", True)
    set_prop(pps, "motion_blur_amount", 0.5 if job["render"].get("motion_blur") else 0.0)
    comp.set_editor_property("post_process_settings", pps)
    binding, keep = _bind(sequence, actor, use_spawnables)
    _transform_keys(binding, n_frames, [cam["location"]], [cam["rotation"]], [1.0, 1.0, 1.0])
    cut_track = add_sequence_track(sequence, unreal.MovieSceneCameraCutTrack)
    cut = cut_track.add_section()
    set_section_range(cut, 0, n_frames)
    cut.set_camera_binding_id(binding_id(sequence, binding))
    return keep


def build_sequence(job: dict, use_spawnables: bool = True):
    """Create (or overwrite) the Level Sequence asset for ``job``.

    Returns ``(sequence, leftover_actors)``. The leftovers are possessable
    actors (only if spawnables are unavailable) that must stay in the level
    until the render finishes."""
    exp = job.get("experiment", "physscene")
    folder = f"{GENERATED}/Sequences/{exp}"
    name = f"LS_{job['job_id']}"
    path = f"{folder}/{name}"
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        unreal.EditorAssetLibrary.delete_asset(path)
    seq = asset_tools().create_asset(name, folder, unreal.LevelSequence, unreal.LevelSequenceFactoryNew())
    fps = int(job["render"]["fps"])
    n = int(job["render"]["num_frames"])
    seq.set_display_rate(unreal.FrameRate(fps, 1))
    seq.set_playback_start(0)
    seq.set_playback_end(n)
    leftovers = []

    for g in job.get("static_geometry", []):
        mats = {0: g["material"]} if g.get("material") else {}
        a = _spawn_mesh_actor(g["name"], g["ue_mesh"], mats, g["scale"], 0, g["location"], g["rotation"])
        b, keep = _bind(seq, a, use_spawnables)
        _transform_keys(b, n, [g["location"]], [g["rotation"]], g["scale"])
        if keep:
            leftovers.append(keep)

    for body in job["bodies"]:
        keys = body["ue_keys"]
        a = _spawn_mesh_actor(
            f"PhysScene_{body['name']}",
            body["ue_mesh"],
            body.get("materials") or {},
            body["scale"],
            int(body["stencil"]),
            keys["location"][0],
            keys["rotation"][0],
        )
        b, keep = _bind(seq, a, use_spawnables)
        _transform_keys(b, n, keys["location"], keys["rotation"], body["scale"], keys.get("visible"))
        if keep:
            leftovers.append(keep)

    keep = _camera(job, seq, n, use_spawnables)
    if keep:
        leftovers.append(keep)
    unreal.EditorAssetLibrary.save_loaded_asset(seq)
    log(f"built {path} ({n} frames @ {fps} fps, {len(job['bodies'])} bodies)")
    return seq, leftovers
