import json

import numpy as np
import yaml

from physscene.cli import main
from physscene.config import ExperimentConfig
from physscene.planner import plan_group
from physscene.validator import validate_states_file

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


def bench_config(tmp_path):
    cfg = {
        "name": "bench",
        "seed": 11,
        "catalog": str(ROOT / "configs" / "catalog_starter.yaml"),
        "levels": [str(ROOT / "configs" / "levels" / "studio_table.yaml")],
        "design": {"events": ["fall"], "objects": ["Ball"], "appearances": {"Ball": ["red"]}, "views": 1},
        "render": {"fps": 24, "num_frames": 48, "resolution": [192, 108], "passes": ["rgb", "mask"]},
        "checks": {"min_object_px": 6},
        "events": {"fall": {"event_time_frac": [0.55, 0.75]}},
        "benchmark": {
            "prefix_frames": 16,
            "valid": {"count": 1, "perturb": {"restitution": {"rel": 0.5}, "friction": {"rel": 0.4}}},
            "invalid": {"types": {"teleport": ["high"], "vanish": ["mid"], "gravity": ["high"], "speed_jump": ["high"]}},
        },
    }
    p = tmp_path / "bench.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_candidates_share_prefix_and_labels(tmp_path):
    cfg = ExperimentConfig.load(bench_config(tmp_path))
    res = plan_group(cfg, "fall", "studio_table", "Ball")
    assert res["ok"], res.get("reason")
    cands = {c.cid: c for c in res["candidates"]}
    assert "ref" in cands and any(c.kind == "valid" for c in cands.values())
    assert any(c.kind == "invalid" for c in cands.values())
    ref = cands["ref"].traj
    for c in cands.values():
        for n in ref.names:
            assert c.traj.positions[n][:17] == ref.positions[n][:17]  # bit-identical prefix (frames 0..16)
        if c.kind != "reference":
            assert c.divergence > 0.02


def test_benchmark_end_to_end_gate(tmp_path):
    cfgp = bench_config(tmp_path)
    out = tmp_path / "run"
    assert main(["run", str(cfgp), "-o", str(out), "--scale", "1.0"]) == 0
    ds = out / "dataset"
    pairs = [json.loads(l) for l in (ds / "benchmark" / "pairs.jsonl").read_text().splitlines()]
    assert {p["label"] for p in pairs} == {"valid", "invalid"}
    for d in ds.rglob("states.json"):
        meta = json.loads((d.parent / "metadata.json").read_text())
        assert validate_states_file(d).valid == (meta["label"] == "valid")
    assert main(["gate", str(out / "plan"), str(ds), "--sample-groups", "1"]) == 0
    report = json.loads((ds / "benchmark" / "gate_report.json").read_text())
    assert report["all_pass"]


def test_violations_on_held_out_alternatives(tmp_path):
    """Violations built from a valid alternative keep that base's hidden
    parameters and are matched to it, not to the reference."""
    cfgp = bench_config(tmp_path)
    d = yaml.safe_load(cfgp.read_text())
    d["benchmark"]["valid"]["count"] = 2
    d["benchmark"]["invalid"]["bases"] = ["f02"]
    d["benchmark"]["invalid"]["types"] = {"teleport": ["high"], "speed_jump": ["high"]}
    cfgp.write_text(yaml.safe_dump(d))
    cfg = ExperimentConfig.load(cfgp)
    assert cfg.benchmark.invalid_bases == ["f02"]
    res = plan_group(cfg, "fall", "studio_table", "Ball")
    assert res["ok"], res.get("reason")
    cands = {c.cid: c for c in res["candidates"]}
    invalid = [c for c in cands.values() if c.kind == "invalid"]
    assert invalid and all(c.base == "f02" and c.cid.startswith("f02_") for c in invalid)
    base = cands["f02"]
    tele = cands["f02_teleport_high"]
    assert tele.params == base.params  # the base's hidden physics, not the reference's
    # a teleport is the base trajectory plus a constant offset after the jump
    off = np.asarray(tele.traj.positions["object"][-1]) - np.asarray(base.traj.positions["object"][-1])
    assert abs(np.linalg.norm(off) - 0.3) < 1e-6
    assert tele.traj.positions["object"][:17] == base.traj.positions["object"][:17]


def test_visibly_different_ignores_changes_the_camera_cannot_see():
    from physscene.benchmark import visibly_different

    n = 10
    ref = {"vis": np.zeros(n), "uv": np.zeros((n, 2)), "inside": np.ones(n, bool)}
    # object hidden behind an occluder in both -> a vanish there is invisible
    assert not visibly_different(ref, {"vis": np.zeros(n), "uv": np.zeros((n, 2))}, px=6)
    seen = {"vis": np.ones(n), "uv": np.zeros((n, 2))}
    gone = {"vis": np.r_[np.ones(5), np.zeros(5)], "uv": np.zeros((n, 2))}
    assert visibly_different(seen, gone, px=6)
    shifted = {"vis": np.ones(n), "uv": np.tile([10.0, 0.0], (n, 1))}
    assert visibly_different(seen, shifted, px=6)
    assert not visibly_different(seen, {"vis": np.ones(n), "uv": np.tile([2.0, 0.0], (n, 1))}, px=6)


def test_teleport_respects_offset_constraint():
    from physscene.spec import Trajectory
    from physscene.violations import teleport

    T = 10
    tr = Trajectory(fps=24, names=["object"], positions={"object": [[0.0, 0.0, 1.0]] * T},
                    quats={"object": [[1.0, 0, 0, 0]] * T})
    out = teleport(tr, 2, 0.3, np.random.default_rng(0), offset_ok=lambda off: off[0] > 0.25)
    assert out.pos("object")[-1][0] > 0.25 and out.pos("object")[0][0] == 0.0


def test_drift_is_smooth_and_sideways():
    from physscene.spec import Trajectory
    from physscene.violations import drift

    T, P, fps = 48, 10, 24
    pos = [[0.5 * f / fps, 0.0, 0.06] for f in range(T)]  # straight roll along +x at 0.5 m/s
    tr = Trajectory(fps=fps, names=["object"], positions={"object": pos}, quats={"object": [[1.0, 0, 0, 0]] * T})
    out = drift(tr, P, 0.09, np.random.default_rng(0))
    p = out.pos("object")
    assert np.allclose(p[: P + 1], np.asarray(pos)[: P + 1])  # untouched prefix
    assert np.allclose(p[:, 0], np.asarray(pos)[:, 0]) and np.allclose(p[:, 2], 0.06)  # only sideways
    t = (T - 1 - P) / fps
    assert abs(abs(p[-1, 1]) - 0.5 * 0.09 * t**2) < 1e-9
    # continuous in position and velocity at onset
    assert abs(p[P + 1, 1]) < 1e-3
    blocked = lambda offs: bool(np.all(offs[:, 1] >= 0))
    assert drift(tr, P, 0.09, np.random.default_rng(1), offsets_ok=blocked).pos("object")[-1, 1] > 0


def test_validator_straight_path_law():
    """A ball rolling straight on a table passes; the same roll with a smooth
    sideways drift is flagged, though it stays well inside the Coulomb bound."""
    from physscene.validator import validate_states

    T, fps, r = 48, 24, 0.06
    pos, quats = [], []
    for f in range(T):
        x = -0.4 + 0.5 * f / fps
        pos.append([x, 0.0, 0.75 + r])
        ang = (x + 0.4) / r  # rolling without slipping about +y
        quats.append([np.cos(ang / 2), 0.0, np.sin(ang / 2), 0.0])
    shapes = {"object": {"type": "sphere", "radius": r}}
    supports = [{"center": [0.0, 0.0, 0.75], "size": [1.6, 0.9], "yaw_deg": 0.0}]
    body = {"name": "object", "dynamic": True}
    params = {"bodies": {"object": {"mass": 0.15, "friction": 0.6}}, "surface": {"friction": 0.45}, "floor": {}}

    def verdict(p):
        tr = {"fps": fps, "names": ["object"], "positions": {"object": p}, "quats": {"object": quats}}
        return validate_states({"layout": {"bodies": [body]}, "params": params, "trajectory": tr}, shapes, supports, 0.0)

    assert "straight_path" not in verdict(pos).summary
    drifted = [[p[0], p[1] + 0.5 * 0.09 * (max(f - 10, 0) / fps) ** 2, p[2]] for f, p in enumerate(pos)]
    assert verdict(drifted).summary.get("straight_path")
