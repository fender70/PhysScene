"""Smoke-test the Unreal driver against a mocked ``unreal`` module.

This cannot check that Unreal accepts the calls; that needs the engine.
It does catch Python errors in the driver's own logic (bad names, wrong
arities, key/channel bookkeeping)."""
import json
import sys
from unittest import mock

from physscene.config import ExperimentConfig
from physscene.planner import plan_experiment


class Channel:
    def __init__(self):
        self.keys = []

    def add_key(self, frame, value, *a):
        self.keys.append((frame, value))


def make_unreal():
    ue = mock.MagicMock(name="unreal")
    ue.EditorAssetLibrary.does_asset_exist.return_value = False

    def new_section():
        sec = mock.MagicMock()
        chans = [Channel() for _ in range(9)]
        sec.get_channels_by_type.return_value = chans
        sec._chans = chans
        return sec

    def new_binding(*_a, **_k):
        b = mock.MagicMock()
        track = mock.MagicMock()
        track.add_section.side_effect = new_section
        b.add_track.return_value = track
        return b

    seq = mock.MagicMock()
    seq.add_spawnable_from_instance.side_effect = new_binding
    ue.AssetToolsHelpers.get_asset_tools.return_value.create_asset.return_value = seq
    ue.FrameNumber.side_effect = lambda f: f
    return ue, seq


def test_driver_builds_sequences_and_configs(tiny_config, tmp_path):
    cfg = ExperimentConfig.load(tiny_config)
    plan_experiment(cfg, tmp_path / "plan")
    ue, seq = make_unreal()
    with mock.patch.dict(sys.modules, {"unreal": ue}):
        for m in [k for k in sys.modules if k.startswith("physscene_ue")]:
            del sys.modules[m]
        from physscene_ue import render, sequence

        jobs = render.load_jobs(str(tmp_path / "plan"), str(tmp_path / "renders"))
        assert len(jobs) == 4
        s, left = sequence.build_sequence(jobs[0])
        assert left == []
        seq.set_playback_end.assert_called_with(30)
        # bodies + static geometry + camera each got a spawnable
        n_expected = len(jobs[0]["bodies"]) + len(jobs[0]["static_geometry"]) + 1
        assert seq.add_spawnable_from_instance.call_count == n_expected
        mj = mock.MagicMock()
        for kind in ("rgb", "labels", "depth"):
            render.configure(mj, jobs[0], kind, str(tmp_path / "out"))
        br = render.BatchRenderer(str(tmp_path / "plan"), str(tmp_path / "renders"), jobs)
        br.pending["abc"] = {"rgb"}
        br.fingerprints["abc"] = "f00"
        od = mock.MagicMock()
        od.get_editor_property.side_effect = lambda k: {
            "job": mock.MagicMock(get_editor_property=lambda _: json.dumps({"job_id": "abc", "kind": "rgb"})),
            "success": True,
        }[k]
        br._on_job_finished(od)
        assert (tmp_path / "renders" / "abc" / "DONE").exists()


def test_fingerprint_matches_between_planner_and_driver():
    from physscene.spec import job_fingerprint

    ue, _ = make_unreal()
    with mock.patch.dict(sys.modules, {"unreal": ue}):
        for m in [k for k in sys.modules if k.startswith("physscene_ue")]:
            del sys.modules[m]
        from physscene_ue import render

        job = {"job_id": "x", "b": [1, 2, {"c": 3.5}]}
        assert render.job_fingerprint(job) == job_fingerprint(job)
