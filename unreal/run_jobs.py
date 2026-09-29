"""Unreal Editor entry point: batch-render a PhysScene plan.

Launched by ``physscene render PLAN --renderer unreal --uproject X.uproject``,
which sets the environment variables below and runs::

    UnrealEditor X.uproject -ExecutePythonScript=<this file> -unattended -RenderOffscreen ...

Environment:
    PHYSSCENE_PLAN            plan directory (required)
    PHYSSCENE_RENDER_ROOT     where raw renders go (default: <plan>/renders)
    PHYSSCENE_JOBS            optional comma-separated job ids
    PHYSSCENE_FORCE=1         re-render jobs that already have a DONE marker
    PHYSSCENE_QUIT_WHEN_DONE=1  close the editor at the end
    PHYSSCENE_POSSESSABLES=1  use level actors instead of spawnables (fallback)

You can also run it from an open editor: Tools > Execute Python Script.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import unreal  # noqa: E402

from physscene_ue import render  # noqa: E402

_handle = None
_ticks = 0


def _wait_then_start(delta_seconds):
    """Start only after the editor has finished loading, because -ExecutePythonScript
    runs before the asset registry is ready."""
    global _handle, _ticks
    _ticks += 1
    registry = unreal.AssetRegistryHelpers.get_asset_registry()
    if registry.is_loading_assets() or _ticks < 30:
        return
    unreal.unregister_slate_post_tick_callback(_handle)
    try:
        render.run_from_env()
    except Exception as e:  # noqa: BLE001
        unreal.log_error(f"[PhysScene] fatal: {e}")
        if os.environ.get("PHYSSCENE_QUIT_WHEN_DONE") == "1":
            unreal.SystemLibrary.quit_editor()
        raise


_handle = unreal.register_slate_post_tick_callback(_wait_then_start)
