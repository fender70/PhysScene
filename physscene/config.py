"""Experiment configuration: which interventions to generate and how to render them."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .catalog import Catalog, Level


@dataclass
class RenderSettings:
    fps: int = 24
    num_frames: int = 72
    resolution: tuple[int, int] = (1280, 720)
    passes: list[str] = field(default_factory=lambda: ["rgb", "mask"])  # rgb, mask, depth
    spatial_samples: int = 1
    temporal_samples: int = 8
    warmup_frames: int = 32
    motion_blur: bool = False
    sensor_width_mm: float = 36.0


@dataclass
class PhysicsSettings:
    backend: str = "mujoco"
    timestep: float = 0.001
    # Extra simulated time before frame 0. The first rendered frame is t=pre_roll.
    # Keep it 0 for CRONOS-style "pushed just before the video starts" clips.
    pre_roll: float = 0.0


@dataclass
class Perturbation:
    """How one physical parameter is resampled for an alternative future.

    ``rel``: multiplicative uniform noise, value * U(1-rel, 1+rel).
    ``abs``: additive uniform noise, value + U(-abs, abs).
    ``range``: replace with U(lo, hi).
    """

    rel: float | None = None
    abs: float | None = None
    range: tuple[float, float] | None = None

    @staticmethod
    def from_dict(d: Any) -> "Perturbation":
        if isinstance(d, (int, float)):
            return Perturbation(rel=float(d))
        return Perturbation(
            rel=d.get("rel"),
            abs=d.get("abs"),
            range=tuple(d["range"]) if "range" in d else None,
        )


@dataclass
class FuturesSettings:
    """Alternative physically plausible futures from identical initial conditions.

    Frame 0 is identical across futures; only the hidden physical parameters
    (and optionally sub-perceptual initial-velocity noise) change."""

    count: int = 0
    perturb: dict[str, Perturbation] = field(default_factory=dict)
    require_same_event: bool = True
    max_attempts: int = 20


@dataclass
class BenchmarkSettings:
    """Physical-validity benchmark: each reference gets matched-prefix candidates.

    * ``prefix_frames``: frames 0..P are identical in every candidate (the
      conditioning prefix). The key event must happen after it.
    * ``valid``: alternative futures that branch at P with resampled hidden
      parameters (no state edits).
    * ``invalid``: controlled violations after P, as ``{type: [severity, ...]}``.
    """

    enabled: bool = False
    prefix_frames: int = 24
    valid_count: int = 2
    valid_perturb: dict[str, Perturbation] = field(default_factory=dict)
    valid_min_divergence: float = 0.02
    valid_max_attempts: int = 20
    invalid: dict[str, list[str]] = field(default_factory=dict)
    invalid_min_divergence: float = 0.03
    # Keep only violations the independent validator confirms (and only valid
    # candidates it accepts), so every label has two independent sources.
    require_validator: bool = True
    # Restrict violation types to events where they are visible,
    # e.g. {gravity: [fall]}. On a table top, weaker gravity looks like lower friction.
    applicable_events: dict[str, list[str]] = field(default_factory=lambda: {"gravity": ["fall"]})
    validator_tol: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict | None) -> "BenchmarkSettings":
        if not d:
            return BenchmarkSettings()
        v = d.get("valid", {}) or {}
        inv = d.get("invalid", {}) or {}
        return BenchmarkSettings(
            enabled=True,
            prefix_frames=int(d.get("prefix_frames", 24)),
            valid_count=int(v.get("count", 2)),
            valid_perturb={k: Perturbation.from_dict(x) for k, x in (v.get("perturb") or {}).items()},
            valid_min_divergence=float(v.get("min_divergence", 0.02)),
            valid_max_attempts=int(v.get("max_attempts", 20)),
            invalid={k: list(x) if isinstance(x, (list, tuple)) else [x] for k, x in (inv.get("types") or {}).items()},
            invalid_min_divergence=float(inv.get("min_divergence", 0.03)),
            require_validator=bool(d.get("require_validator", True)),
            applicable_events=dict(inv.get("applicable_events", {"gravity": ["fall"]})),
            validator_tol=dict(d.get("validator_tol", {}) or {}),
        )


@dataclass
class DesignSettings:
    events: list[str] = field(default_factory=lambda: ["fall", "collision", "occlusion"])
    scenes: list[str] = field(default_factory=list)  # level names; empty = all levels
    objects: list[str] = field(default_factory=list)  # empty = all catalog objects
    appearances: str | dict[str, list[str]] = "all"
    views: int = 3
    # "full" = full factorial grid; an int N = N random cells of the grid.
    sampling: str | int = "full"
    max_layout_attempts: int = 60


@dataclass
class ExperimentConfig:
    name: str
    seed: int
    catalog: Catalog
    levels: dict[str, Level]
    design: DesignSettings
    render: RenderSettings
    physics: PhysicsSettings
    futures: FuturesSettings
    events: dict[str, dict[str, Any]]  # per-event template parameters
    checks: dict[str, Any]
    output_layout: str = "cronos"
    benchmark: BenchmarkSettings = field(default_factory=BenchmarkSettings)
    source_path: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def load(path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        d = yaml.safe_load(path.read_text())
        base = path.parent

        def rel(p: str) -> Path:
            q = Path(p)
            return q if q.is_absolute() else (base / q)

        catalog = Catalog.load(rel(d["catalog"]))
        levels = {}
        for lp in d["levels"]:
            lvl = Level.load(rel(lp))
            levels[lvl.name] = lvl

        dz = d.get("design", {})
        design = DesignSettings(
            events=list(dz.get("events", ["fall", "collision", "occlusion"])),
            scenes=list(dz.get("scenes", [])) or list(levels),
            objects=list(dz.get("objects", [])) or list(catalog.objects),
            appearances=dz.get("appearances", "all"),
            views=int(dz.get("views", 3)),
            sampling=dz.get("sampling", "full"),
            max_layout_attempts=int(dz.get("max_layout_attempts", 60)),
        )
        rz = d.get("render", {})
        render = RenderSettings(
            fps=int(rz.get("fps", 24)),
            num_frames=int(rz.get("num_frames", 72)),
            resolution=tuple(rz.get("resolution", (1280, 720))),
            passes=list(rz.get("passes", ["rgb", "mask"])),
            spatial_samples=int(rz.get("spatial_samples", 1)),
            temporal_samples=int(rz.get("temporal_samples", 8)),
            warmup_frames=int(rz.get("warmup_frames", 32)),
            motion_blur=bool(rz.get("motion_blur", False)),
            sensor_width_mm=float(rz.get("sensor_width_mm", 36.0)),
        )
        pz = d.get("physics", {})
        physics = PhysicsSettings(
            backend=pz.get("backend", "mujoco"),
            timestep=float(pz.get("timestep", 0.001)),
            pre_roll=float(pz.get("pre_roll", 0.0)),
        )
        fz = d.get("futures", {}) or {}
        futures = FuturesSettings(
            count=int(fz.get("count", 0)),
            perturb={k: Perturbation.from_dict(v) for k, v in (fz.get("perturb") or {}).items()},
            require_same_event=bool(fz.get("require_same_event", True)),
            max_attempts=int(fz.get("max_attempts", 20)),
        )
        for s in design.scenes:
            if s not in levels:
                raise ValueError(f"design.scenes references unknown level {s!r}")
        for o in design.objects:
            if o not in catalog.objects:
                raise ValueError(f"design.objects references unknown object {o!r}")
        return ExperimentConfig(
            name=d.get("name", path.stem),
            seed=int(d.get("seed", 0)),
            catalog=catalog,
            levels=levels,
            design=design,
            render=render,
            physics=physics,
            futures=futures,
            events=d.get("events", {}) or {},
            checks=d.get("checks", {}) or {},
            output_layout=d.get("output_layout", "cronos"),
            benchmark=BenchmarkSettings.from_dict(d.get("benchmark")),
            source_path=path,
            raw=d,
        )

    def appearances_for(self, obj_key: str) -> list[str]:
        asset = self.catalog.objects[obj_key]
        if self.design.appearances == "all":
            return list(asset.appearances)
        if isinstance(self.design.appearances, dict):
            return list(self.design.appearances.get(obj_key, asset.appearances))
        return list(asset.appearances)[:1]
