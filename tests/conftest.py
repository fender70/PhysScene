import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "unreal"))


@pytest.fixture
def tiny_config(tmp_path):
    """A 2-group, 1-view, short-clip experiment built on the starter assets."""
    cfg = {
        "name": "tiny",
        "seed": 3,
        "catalog": str(ROOT / "configs" / "catalog_starter.yaml"),
        "levels": [str(ROOT / "configs" / "levels" / "studio_table.yaml")],
        "design": {"events": ["fall", "collision"], "objects": ["Ball"], "appearances": {"Ball": ["red"]}, "views": 1},
        "render": {"fps": 12, "num_frames": 30, "resolution": [160, 96], "passes": ["rgb", "mask", "depth"]},
        "futures": {"count": 1, "perturb": {"friction": {"rel": 0.2}, "initial_speed": {"rel": 0.05}}},
    }
    p = tmp_path / "tiny.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p
