"""Small shims over Unreal Python APIs that were renamed across UE 5.x releases.

Every helper tries the current API first and falls back to older names, so
the driver works from UE 5.3 through 5.7. If Epic renames something again,
this is the only file to touch.
"""
from __future__ import annotations

import unreal


def log(msg: str) -> None:
    unreal.log(f"[PhysScene] {msg}")


def warn(msg: str) -> None:
    unreal.log_warning(f"[PhysScene] {msg}")


def actor_subsystem():
    return unreal.get_editor_subsystem(unreal.EditorActorSubsystem)


def level_subsystem():
    return unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)


def asset_tools():
    return unreal.AssetToolsHelpers.get_asset_tools()


def add_sequence_track(sequence, track_class):
    """``add_track`` (5.2+) or ``add_master_track`` (<=5.1)."""
    if hasattr(sequence, "add_track"):
        return sequence.add_track(track_class)
    return sequence.add_master_track(track_class)


def double_channels(section):
    """Float channels of a transform section in order:
    Location X Y Z, Rotation X(roll) Y(pitch) Z(yaw), Scale X Y Z."""
    for ch_type in ("MovieSceneScriptingDoubleChannel", "MovieSceneScriptingFloatChannel"):
        t = getattr(unreal, ch_type, None)
        if t is not None and hasattr(section, "get_channels_by_type"):
            chans = section.get_channels_by_type(t)
            if chans:
                return list(chans)
    return list(section.get_all_channels())


def set_section_range(section, start: int, end: int) -> None:
    if hasattr(section, "set_range"):
        section.set_range(start, end)
    else:
        section.set_start_frame(start)
        section.set_end_frame(end)


def add_key(channel, frame: int, value: float, linear: bool = True) -> None:
    interp = unreal.MovieSceneKeyInterpolation.LINEAR if linear else unreal.MovieSceneKeyInterpolation.AUTO
    channel.add_key(unreal.FrameNumber(frame), float(value), 0.0, unreal.MovieSceneTimeUnit.DISPLAY_RATE, interp)


def binding_id(sequence, binding):
    """Object binding id used by camera-cut sections."""
    lib = unreal.MovieSceneSequenceExtensions
    for fn in ("get_portable_binding_id", "make_binding_id"):
        f = getattr(lib, fn, None)
        if f is not None:
            try:
                return f(sequence, binding)
            except TypeError:
                try:
                    return f(sequence, binding, unreal.MovieSceneObjectBindingSpace.LOCAL)
                except Exception:  # noqa: BLE001
                    pass
    if hasattr(sequence, "get_binding_id"):
        return sequence.get_binding_id(binding)
    bid = unreal.MovieSceneObjectBindingID()
    bid.set_editor_property("guid", binding.get_id())
    return bid


def add_spawnable(sequence, actor):
    """Turn ``actor`` into a spawnable owned by the sequence. Returns the binding
    or ``None`` if this engine version refuses (the caller then uses a possessable)."""
    for obj in (sequence, unreal.MovieSceneSequenceExtensions):
        fn = getattr(obj, "add_spawnable_from_instance", None)
        if fn is None:
            continue
        try:
            return fn(actor) if obj is sequence else fn(sequence, actor)
        except Exception as e:  # noqa: BLE001
            warn(f"add_spawnable_from_instance failed ({e}); falling back to possessable")
    return None


def set_prop(obj, name: str, value) -> bool:
    try:
        obj.set_editor_property(name, value)
        return True
    except Exception:  # noqa: BLE001
        return False


def enum_value(enum_cls, *names):
    for n in names:
        if hasattr(enum_cls, n):
            return getattr(enum_cls, n)
    raise AttributeError(f"{enum_cls} has none of {names}")
