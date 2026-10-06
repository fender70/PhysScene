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
