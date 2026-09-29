import json

import numpy as np

from physscene.cli import main
from physscene.config import ExperimentConfig
from physscene.planner import plan_experiment, plan_group
from physscene.spec import Trajectory


def test_group_plans_and_futures_share_first_frame(tiny_config):
    cfg = ExperimentConfig.load(tiny_config)
    res = plan_group(cfg, "fall", "studio_table", "Ball")
    assert res["ok"], res.get("reason")
    assert len(res["futures"]) == 1
    base = np.asarray(res["traj"].positions["object"])
    fut = np.asarray(res["futures"][0]["traj"].positions["object"])
    assert np.allclose(base[0], fut[0])  # identical initial conditions
    assert not np.allclose(base[-1], fut[-1])  # but a different future


def test_planning_is_deterministic(tiny_config, tmp_path):
    cfg = ExperimentConfig.load(tiny_config)
    a = plan_experiment(cfg, tmp_path / "a")
    b = plan_experiment(cfg, tmp_path / "b")
    assert a["num_jobs"] == b["num_jobs"] == 2 * 2  # 2 groups x (1 + 1 future)
    ja = sorted(p.name for p in (tmp_path / "a" / "jobs").glob("*.json"))
    jb = sorted(p.name for p in (tmp_path / "b" / "jobs").glob("*.json"))
    assert ja == jb
    for name in ja:
        assert (tmp_path / "a" / "jobs" / name).read_text() == (tmp_path / "b" / "jobs" / name).read_text()


def test_end_to_end_preview(tiny_config, tmp_path):
    out = tmp_path / "run"
    rc = main(["run", str(tiny_config), "-o", str(out), "--renderer", "preview", "--scale", "1.0"])
    assert rc == 0
    ds = out / "dataset"
    samples = sorted(ds.rglob("metadata.json"))
    assert len(samples) == 4
    d = ds / "fall" / "studio_table" / "Ball" / "red" / "v0"
    meta = json.loads((d / "metadata.json").read_text())
    assert meta["event"] == "fall" and meta["object"] == "Ball"
    assert (d / "movies" / "complete.mp4").stat().st_size > 0
    z = np.load(d / "masks.npz")
    assert z["masks"].shape[0] == 30 and z["masks"][0, ..., 0].sum() > 0
    cam = json.loads((d / "camera.json").read_text())
    # object centroid from mask agrees with projected physics position
    states = json.loads((d / "states.json").read_text())
    traj = Trajectory.from_dict(states["trajectory"])
    k = np.asarray(cam["intrinsics"])
    w2c = np.asarray(cam["world_to_cam_opencv"])
    p = np.append(traj.pos("object")[0], 1.0)
    pc = (w2c @ p)[:3]
    uv = (k @ pc)[:2] / pc[2]
    ys, xs = np.nonzero(z["masks"][0, ..., 0])
    assert np.hypot(xs.mean() - uv[0], ys.mean() - uv[1]) < 3.0
    assert (d / "alternatives" / "f01" / "metadata.json").exists()
    assert (ds / "cronos_config.json").exists()
