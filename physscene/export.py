"""Convert raw renders into a benchmark dataset.

Output layout (compatible with the CRONOS evaluation code, which reads
``metadata.json``, ``mask/frame_0000.jpg`` and ``movies/complete.mp4``)::

    <dataset>/<event>/<scene>/<object>/<appearance>/<view>/
        movies/complete.mp4        full clip (H.264)
        rgb/frame_0000.png ...     lossless frames
        mask/frame_0000.jpg ...    one channel per masked body (R=object, G=collider/occluder)
        masks.npz                  lossless boolean masks [T, H, W, N] + body names
        depth/frame_0000.png ...   uint16 depth in millimetres (if rendered)
        metadata.json              factors, prompts, physics, event frame, provenance
        camera.json                intrinsics + extrinsics (OpenCV and Unreal conventions)
        states.json                simulator state: layout, physical params, per-frame poses
        annotations.json           per-frame 2D boxes, centroids and visible areas from masks
        alternatives/f01/...       alternative futures (same frame 0), same structure
"""
from __future__ import annotations

import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .geometry import quat_to_matrix
from .spec import job_fingerprint

log = logging.getLogger("physscene.export")
IMG_EXT = (".png", ".jpg", ".jpeg", ".exr", ".npy")


def _frames(d: Path, prefer: str | None = None, avoid: str | None = "FinalImage") -> list[Path]:
    """Sorted frame files under ``d`` (recursively). Movie Render Queue writes
    ``<pass name>/<frame>.<ext>``, so pick the right pass sub-directory."""
    if not d.exists():
        return []
    files = [p for p in d.rglob("*") if p.suffix.lower() in IMG_EXT and p.is_file()]
    by_dir: dict[Path, list[Path]] = defaultdict(list)
    for f in files:
        by_dir[f.parent].append(f)
    if not by_dir:
        return []
    dirs = list(by_dir)
    if prefer:
        pref = [x for x in dirs if prefer.lower() in x.name.lower()]
        if pref:
            dirs = pref
    if avoid and len(dirs) > 1:
        dirs = [x for x in dirs if avoid.lower() not in x.name.lower()] or dirs
    chosen = sorted(dirs, key=lambda x: -len(by_dir[x]))[0]
    return sorted(by_dir[chosen], key=lambda p: p.name)


def _read_image(p: Path) -> np.ndarray:
    if p.suffix == ".npy":
        return np.load(p)
    if p.suffix.lower() == ".exr":
        return _read_exr(p)
    return np.asarray(Image.open(p))


def _read_exr(p: Path) -> np.ndarray:
    try:
        import OpenEXR  # type: ignore
        import Imath  # type: ignore

        f = OpenEXR.InputFile(str(p))
        dw = f.header()["dataWindow"]
        w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
        ch = "R" if "R" in f.header()["channels"] else sorted(f.header()["channels"])[0]
        buf = f.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT))
        return np.frombuffer(buf, dtype=np.float32).reshape(h, w)
    except ImportError:
        pass
    import imageio.v3 as iio

    return np.asarray(iio.imread(p))


def decode_stencil_bits(img: np.ndarray) -> np.ndarray:
    """RGB-bit encoded stencil (R=bit0, G=bit1, B=bit2) -> integer ids 0..7."""
    img = np.asarray(img)
    if img.ndim == 2:
        return (img > 127).astype(np.uint8)
    if img.dtype != np.uint8:  # float render targets
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    bits = (img[..., :3] > 127).astype(np.uint8)
    return bits[..., 0] | (bits[..., 1] << 1) | (bits[..., 2] << 2)


def write_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    # H.264 with yuv420p needs even dimensions
    frames = [f[: h - h % 2, : w - w % 2, :3] for f in frames]
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=1) as wr:
        for f in frames:
            wr.append_data(f)


def camera_json(job: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    cam = job["camera"]
    sx = width / cam["resolution"][0]
    k = np.asarray(cam["intrinsics"]) * np.array([[sx], [sx], [1]])
    k[2, 2] = 1.0
    r_c2w = quat_to_matrix(cam["quat"])  # columns: fwd, left, up (PhysScene camera axes)
    # OpenCV camera axes: x right, y down, z forward
    cv_axes = np.stack([-r_c2w[:, 1], -r_c2w[:, 2], r_c2w[:, 0]], axis=1)
    c2w = np.eye(4)
    c2w[:3, :3] = cv_axes
    c2w[:3, 3] = cam["position"]
    return {
        "resolution": [width, height],
        "intrinsics": k.tolist(),
        "hfov_deg": cam["hfov_deg"],
        "cam_to_world_opencv": c2w.tolist(),
        "world_to_cam_opencv": np.linalg.inv(c2w).tolist(),
        "world_frame": "right-handed, Z-up, metres",
        "position": cam["position"],
        "quat_wxyz": cam["quat"],
        "unreal": cam["ue"],
    }


def annotations(masks: np.ndarray, names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"bodies": names, "frames": []}
    for t in range(masks.shape[0]):
        fr = {}
        for i, n in enumerate(names):
            m = masks[t, ..., i]
            area = int(m.sum())
            if area == 0:
                fr[n] = {"area": 0, "bbox": None, "centroid": None}
                continue
            ys, xs = np.nonzero(m)
            fr[n] = {
                "area": area,
                "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "centroid": [float(xs.mean()), float(ys.mean())],
            }
        out["frames"].append(fr)
    return out


def export_job(job: dict[str, Any], plan_dir: Path, render_dir: Path, dataset_dir: Path, keep_frames: bool = True) -> Path:
    dst = dataset_dir / job["output_rel"]
    dst.mkdir(parents=True, exist_ok=True)
    fps = job["render"]["fps"]

    rgb_files = _frames(render_dir / "rgb", prefer="FinalImage", avoid=None)
    if not rgb_files:
        raise FileNotFoundError(f"no RGB frames in {render_dir / 'rgb'}")
    rgb = [np.asarray(Image.open(p).convert("RGB")) for p in rgb_files]
    h, w = rgb[0].shape[:2]
    write_video(rgb, dst / "movies" / "complete.mp4", fps)
    if keep_frames:
        (dst / "rgb").mkdir(exist_ok=True)
        for i, p in enumerate(rgb_files):
            if p.suffix == ".png":
                shutil.copyfile(p, dst / "rgb" / f"frame_{i:04d}.png")
            else:
                Image.fromarray(rgb[i]).save(dst / "rgb" / f"frame_{i:04d}.png")

    masked = sorted([b for b in job["bodies"] if b["stencil"] > 0], key=lambda b: b["stencil"])
    names = [b["name"] for b in masked]
    label_files = _frames(render_dir / "labels")
    if label_files:
        ids = np.stack([decode_stencil_bits(_read_image(p)) for p in label_files])
        if ids.shape[1:] != (h, w):
            ids = np.stack([np.asarray(Image.fromarray(x).resize((w, h), Image.NEAREST)) for x in ids])
        masks = np.stack([ids == b["stencil"] for b in masked], axis=-1)  # [T,H,W,N]
        np.savez_compressed(dst / "masks.npz", masks=masks, names=np.array(names))
        (dst / "mask").mkdir(exist_ok=True)
        for t in range(masks.shape[0]):
            img = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(min(3, len(names))):
                img[..., i] = masks[t, ..., i] * 255
            if len(names) == 1:
                img = img[..., 0]  # single object -> grayscale, as CRONOS does
            Image.fromarray(img).save(dst / "mask" / f"frame_{t:04d}.jpg", quality=95)
        (dst / "annotations.json").write_text(json.dumps(annotations(masks, names)))

    depth_files = _frames(render_dir / "depth")
    if depth_files:
        (dst / "depth").mkdir(exist_ok=True)
        for t, p in enumerate(depth_files):
            d = _read_image(p).astype(np.float32)
            if d.ndim == 3:
                d = d[..., 0]
            if p.suffix.lower() == ".exr" and job["render"].get("depth_units", "m") == "cm":
                d = d / 100.0
            mm = np.clip(np.nan_to_num(d, posinf=0.0) * 1000.0, 0, 65535).astype(np.uint16)
            if mm.shape != (h, w):
                mm = np.asarray(Image.fromarray(mm).resize((w, h), Image.NEAREST))
            Image.fromarray(mm).save(dst / "depth" / f"frame_{t:04d}.png")

    meta = dict(job["metadata"])
    meta.update({"job_id": job["job_id"], "num_frames": len(rgb_files), "resolution": [w, h], "mask_bodies": names})
    (dst / "metadata.json").write_text(json.dumps(meta, indent=1))
    (dst / "camera.json").write_text(json.dumps(camera_json(job, w, h), indent=1))
    tdir = (plan_dir / job["trajectory"]).parent
    states = {
        "layout": json.loads((tdir / "layout.json").read_text()),
        "params": json.loads((tdir / "params.json").read_text()),
        "trajectory": json.loads((plan_dir / job["trajectory"]).read_text()),
        "unreal_keys": {b["name"]: b["ue_keys"] for b in job["bodies"]},
    }
    (dst / "states.json").write_text(json.dumps(states))
    return dst


def cronos_prompt_config(dataset_dir: Path) -> dict[str, Any]:
    """Collect the prompt dictionaries that the CRONOS evaluation ``config.py`` expects."""
    surf: dict[str, dict[str, str]] = defaultdict(dict)
    extra: dict[str, dict[str, str]] = defaultdict(dict)
    objects: dict[str, str] = {}
    apps: dict[str, list[str]] = defaultdict(list)
    for mp in dataset_dir.rglob("metadata.json"):
        if "alternatives" in mp.parts:
            continue
        m = json.loads(mp.read_text())
        surf[m["event"]][m["scene"]] = m.get("surface_name", "")
        for role in ("collider_name", "occluder_name"):
            if role in m:
                extra[m["event"]][m["scene"]] = m[role]
        objects[m["object"]] = m.get("object_name", m["object"])
        a = m.get("appearance_name", m.get("appearance"))
        if a not in apps[m["object"]]:
            apps[m["object"]].append(a)
    return {
        "SURFACE_DICT": dict(surf),
        "EXTRA_ELEMENTS": dict(extra),
        "OBJECT_NAMES": objects,
        "OBJECT_APPEARANCES": dict(apps),
    }


def export_plan(plan_dir: str | Path, dataset_dir: str | Path, render_root: str | Path | None = None, keep_frames: bool = True) -> list[Path]:
    plan_dir = Path(plan_dir)
    dataset_dir = Path(dataset_dir)
    render_root = Path(render_root) if render_root else plan_dir / "renders"
    out = []
    missing = []
    for jp in sorted((plan_dir / "jobs").glob("*.json")):
        job = json.loads(jp.read_text())
        rdir = render_root / job["job_id"]
        done = rdir / "DONE"
        if not done.exists() or done.read_text().split()[:1] != [job_fingerprint(job)]:
            missing.append(job["output_rel"])
            continue
        out.append(export_job(job, plan_dir, rdir, dataset_dir, keep_frames))
    if missing:
        log.warning("%d jobs have no up-to-date renders (e.g. %s); skipped", len(missing), missing[0])
    (dataset_dir / "cronos_config.json").write_text(json.dumps(cronos_prompt_config(dataset_dir), indent=1))
    shutil.copyfile(plan_dir / "plan.json", dataset_dir / "plan.json")
    return out
