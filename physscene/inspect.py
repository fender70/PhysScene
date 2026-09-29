"""Plan inspection: text summary and top-down plots of every physics group."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np


def inspect_plan(plan_dir: Path, out_dir: Path | None = None) -> None:
    plan = json.loads((plan_dir / "plan.json").read_text())
    groups = plan["groups"]
    ok = [g for g in groups if g["ok"]]
    print(f"experiment {plan['experiment']} (seed {plan['seed']}): {plan['num_jobs']} jobs, {len(ok)}/{len(groups)} groups ok")
    print("  attempts per group:", Counter(g["attempts"] for g in ok).most_common())
    for g in groups:
        if not g["ok"]:
            print(f"  FAILED {g['event']}/{g['scene']}/{g['obj']}: {g['reason']}")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return
    out_dir = out_dir or plan_dir / "inspect"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = plan["config"]
    base = Path(plan["config_path"]).parent if plan.get("config_path") else plan_dir
    from .catalog import Level

    levels = {lv.name: lv for lv in (Level.load(base / p) for p in cfg["levels"])}
    for g in ok:
        gdir = plan_dir / "groups" / g["group_id"]
        layout = json.loads((gdir / "layout.json").read_text())
        lvl = levels[layout["level"]]
        fig, ax = plt.subplots(figsize=(6, 6))
        for s in lvl.surfaces.values():
            c = [s.to_world((sx * s.size[0] / 2, sy * s.size[1] / 2))[:2] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            ax.add_patch(Polygon(c, closed=True, fc="#d8c3a5", ec="#8e7a5b", alpha=0.6))
        trajs = [gdir / "trajectory.json"] + sorted(gdir.glob("futures/*/trajectory.json"))
        for i, tp in enumerate(trajs):
            tr = json.loads(tp.read_text())
            for name in tr["names"]:
                p = np.asarray(tr["positions"][name])
                style = "-" if i == 0 else ":"
                color = "C0" if name == "object" else "C3"
                ax.plot(p[:, 0], p[:, 1], style, color=color, lw=2 if i == 0 else 1, label=f"{name}" if i == 0 else None)
                ax.plot(*p[0, :2], "o", color=color)
        for k, cam in enumerate(json.loads((gdir / "cameras.json").read_text())):
            pos, tgt = np.asarray(cam["position"]), np.asarray(cam["target"])
            ax.annotate("", xy=tgt[:2], xytext=pos[:2], arrowprops=dict(arrowstyle="->", color="C2"))
            ax.text(pos[0], pos[1], f"v{k}", color="C2")
        ax.set_aspect("equal")
        ax.set_title(f"{g['event']} / {g['scene']} / {g['obj']}  (dotted = futures)")
        ax.legend(loc="upper right", fontsize=8)
        fig.savefig(out_dir / f"{g['event']}_{g['scene']}_{g['obj']}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    print(f"plots -> {out_dir}")
