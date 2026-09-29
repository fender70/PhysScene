"""Write a PhysScene level manifest from actors tagged in the Unreal editor.

This is the one-time "annotation" of a level. Instead of hand-placing objects
for every scene (as in CRONOS), you tag the relevant furniture once:

* ``PhysSceneSurface``: a support surface (table top, counter, floor patch).
  The top face of the mesh's bounding box becomes the surface.
  Optional tags: ``PhysScenePrompt=wooden table``, ``PhysSceneFriction=0.4``,
  ``PhysSceneFloorLevel`` (not elevated: objects cannot fall off).
* ``PhysSceneCollider``: static obstacles the physics should know about
  (walls, chair legs...), exported as boxes.
* ``PhysSceneCameraBounds``: one actor (e.g. a box volume) whose bounds the
  camera must stay inside.
* ``PhysSceneHide``: actors to hide while rendering.

Usage (editor Python console)::

    import physscene_ue.level_export as le
    le.export_level("C:/data/levels/kitchen.yaml", name="kitchen")

The output is JSON, which is also valid YAML, so ``physscene`` loads it directly.
"""
from __future__ import annotations

import json

import unreal

from .ue_compat import actor_subsystem, log


def _tags(actor) -> list[str]:
    return [str(t) for t in actor.tags]


def _tag_value(tags, key, default=None):
    for t in tags:
        if t.startswith(key + "="):
            return t.split("=", 1)[1]
    return default


def _rh(v) -> list[float]:
    """Unreal cm (left-handed) -> metres right-handed."""
    return [v.x / 100.0, -v.y / 100.0, v.z / 100.0]


def _local_box(actor):
    comp = actor.get_component_by_class(unreal.StaticMeshComponent)
    if comp is None:
        origin, extent = actor.get_actor_bounds(False)
        return origin, extent, 0.0
    lo, hi = comp.get_local_bounds()
    xf = comp.get_world_transform()
    center_local = unreal.Vector((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, (lo.z + hi.z) / 2)
    center = xf.transform_location(center_local)
    s = xf.scale3d
    extent = unreal.Vector(abs(hi.x - lo.x) / 2 * s.x, abs(hi.y - lo.y) / 2 * s.y, abs(hi.z - lo.z) / 2 * s.z)
    yaw = xf.rotation.rotator().yaw
    return center, extent, yaw


def export_level(path: str, name: str | None = None, floor_z: float | None = None) -> dict:
    world = unreal.EditorLevelLibrary.get_editor_world()
    level_path = world.get_path_name().split(".")[0]
    name = name or level_path.rsplit("/", 1)[-1]
    surfaces, colliders, hide = {}, [], []
    cam_bounds = None
    lowest = None
    for a in actor_subsystem().get_all_level_actors():
        tags = _tags(a)
        label = a.get_actor_label()
        if "PhysSceneHide" in tags:
            hide.append(label)
        if "PhysSceneSurface" in tags:
            c, e, yaw = _local_box(a)
            top = c.z + e.z
            key = label.replace(" ", "_")
            surfaces[key] = {
                "center": _rh(unreal.Vector(c.x, c.y, top)),
                "size": [2 * e.x / 100.0, 2 * e.y / 100.0],
                "yaw_deg": -yaw,  # handedness flip
                "thickness": min(2 * e.z / 100.0, 0.1),
                "friction": float(_tag_value(tags, "PhysSceneFriction", 0.5)),
                "prompt_name": _tag_value(tags, "PhysScenePrompt", label),
            }
            if "PhysSceneFloorLevel" in tags:
                surfaces[key]["elevated"] = False
                lowest = top / 100.0 if lowest is None else min(lowest, top / 100.0)
        if "PhysSceneCollider" in tags:
            c, e, yaw = _local_box(a)
            colliders.append(
                {
                    "name": label,
                    "shape": {"type": "box", "half_extents": [e.x / 100.0, e.y / 100.0, e.z / 100.0]},
                    "center": _rh(c),
                    "yaw_deg": -yaw,
                }
            )
        if "PhysSceneCameraBounds" in tags:
            o, e = a.get_actor_bounds(False)
            lo = _rh(unreal.Vector(o.x - e.x, o.y + e.y, o.z - e.z))
            hi = _rh(unreal.Vector(o.x + e.x, o.y - e.y, o.z + e.z))
            cam_bounds = (lo, hi)
    manifest = {
        "name": name,
        "ue_map": level_path,
        "floor_z": floor_z if floor_z is not None else (lowest if lowest is not None else 0.0),
        "surfaces": surfaces,
        "colliders": colliders,
        "hide_actors": hide,
        "camera": {"distance": [0.8, 3.5], "elevation_deg": [10, 40], "hfov_deg": [45, 60]},
    }
    if cam_bounds:
        manifest["camera"]["bounds_min"], manifest["camera"]["bounds_max"] = cam_bounds
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=1)
    log(f"wrote level manifest {path}: {len(surfaces)} surfaces, {len(colliders)} colliders")
    return manifest
