"""Launch Unreal Editor to render a plan with the PhysScene UE driver."""
from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

UE_SCRIPT = Path(__file__).resolve().parent.parent / "unreal" / "run_jobs.py"


def find_editor(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("UE_EDITOR")
    if env:
        return env
    candidates = [
        r"C:\Program Files\Epic Games\UE_5.7\Engine\Binaries\Win64\UnrealEditor.exe",
        "/Users/Shared/Epic Games/UE_5.7/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor",
        os.path.expanduser("~/UnrealEngine/Engine/Binaries/Linux/UnrealEditor"),
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    raise FileNotFoundError("Unreal Editor not found; pass --editor or set UE_EDITOR")


def build_command(
    plan_dir: str | Path,
    uproject: str | Path,
    editor: str | None = None,
    jobs: list[str] | None = None,
    render_root: str | Path | None = None,
    extra_args: list[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Command line + environment for a batch render.

    The full editor (not ``-Cmd``) is used because Movie Render Queue's PIE
    executor needs the editor loop to tick. ``-RenderOffscreen`` keeps it
    windowless on Linux and Windows."""
    plan_dir = Path(plan_dir).resolve()
    env = dict(os.environ)
    env["PHYSSCENE_PLAN"] = str(plan_dir)
    env["PHYSSCENE_RENDER_ROOT"] = str(Path(render_root).resolve() if render_root else plan_dir / "renders")
    env["PHYSSCENE_QUIT_WHEN_DONE"] = "1"
    if jobs:
        env["PHYSSCENE_JOBS"] = ",".join(jobs)
    cmd = [
        find_editor(editor),
        str(Path(uproject).resolve()),
        f"-ExecutePythonScript={UE_SCRIPT}",
        "-unattended",
        "-nosplash",
        "-nopause",
        "-stdout",
        "-FullStdOutLogOutput",
        "-RenderOffscreen",
        "-log",
    ] + list(extra_args or [])
    return cmd, env


def render_with_unreal(plan_dir, uproject, editor=None, jobs=None, render_root=None, dry_run=False, extra_args=None) -> int:
    cmd, env = build_command(plan_dir, uproject, editor, jobs, render_root, extra_args)
    printable = " ".join(shlex.quote(c) for c in cmd)
    envs = " ".join(f"{k}={shlex.quote(env[k])}" for k in env if k.startswith("PHYSSCENE_"))
    print(f"{envs} {printable}")
    if dry_run:
        return 0
    return subprocess.call(cmd, env=env)
