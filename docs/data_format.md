# Data format

## Plan directory (`physscene plan`)

```
plan/
  plan.json                     config snapshot, per-group status and attempts
  manifest.jsonl                one line per render job (job_id, factors, output_rel)
  groups/<group_id>/            one physics group = (event, scene, object)
    layout.json                 initial conditions (bodies, poses, velocities)
    params.json                 physical parameters used
    trajectory.json             per-frame poses, velocities and contacts
    cameras.json                one camera per view
    check.json                  physics and view check results
    futures/f01/...             alternative futures (layout, params, trajectory, applied perturbations)
  jobs/<job_id>.json            self-contained render job (Unreal keys, camera, materials, metadata)
  renders/<job_id>/             raw renders (rgb/, labels/, depth/, DONE)
```

## Dataset directory (`physscene export`)

```
dataset/<event>/<scene>/<object>/<appearance>/<view>/
  movies/complete.mp4
  rgb/frame_0000.png ...
  mask/frame_0000.jpg ...       R = object, G = collider/occluder (grayscale if only one body)
  masks.npz                     masks: bool [T, H, W, N]; names: [N]
  depth/frame_0000.png ...      uint16, millimetres (0 = sky / no hit)
  metadata.json
  camera.json
  states.json
  annotations.json
  alternatives/f01/ ...         same structure, same frame 0, different hidden physics
  violations/<type>_<severity>/ physically invalid candidate (benchmark mode), same structure
dataset/benchmark/              pairs.jsonl, summary.json, frozen_manifest.json, GATE_REPORT.md (benchmark mode)
dataset/cronos_config.json      SURFACE_DICT / EXTRA_ELEMENTS / OBJECT_NAMES / OBJECT_APPEARANCES
dataset/plan.json
```

### `metadata.json`

The keys `event`, `scene` and `object` are what the CRONOS evaluation code
requires. PhysScene adds `appearance`, `view`, `future`, `surface`, prompt
fields (`object_name`, `appearance_name`, `surface_name`, `collider_name` /
`occluder_name`), `fps`, `num_frames`, `resolution`, `event_frame` (the
frame where the key moment happens), `physics` (all parameters),
`mask_bodies`, `seed`, `group_id` and `job_id`.

In benchmark mode, `candidate` holds `{id, kind: reference|valid|invalid,
prefix_frames, branch_frame, violation: {type, severity, value, onset_frame},
divergence_m, applied (resampled parameters), validator (independent verdict)}`,
and `label` is `valid` or `invalid`. `provenance` records the PhysScene
version and commit, the config and input SHA-256, the MuJoCo version,
the job fingerprint, the renderer and the render settings.

### `camera.json`

* `intrinsics`: 3×3 pinhole K at the exported resolution.
* `cam_to_world_opencv` / `world_to_cam_opencv`: 4×4, OpenCV camera axes
  (x right, y down, z forward).
* World frame: right-handed, Z-up, metres. The Unreal equivalents are under `unreal`.

### `states.json`

* `layout`: initial state of every body (collision-primitive pose, velocity).
* `params`: mass, friction, restitution, rolling friction and damping per body,
  plus surface and floor.
* `trajectory`: per-frame positions, quaternions (wxyz), linear velocities and
  contact pairs.
* `unreal_keys`: per-frame actor location (cm) and rotation (roll, pitch, yaw
  in degrees), exactly as rendered. Frames where a body is invisible (`vanish`)
  park it far below the level.
* `shapes`, `world`: collision primitives and support geometry, so the state
  can be validated without the plan (`physscene.validator.validate_states_file`).
* `trajectory.visible` (optional): per-frame visibility per body.

## Coordinate conventions

PhysScene uses a right-handed, Z-up, metre frame. Unreal uses a left-handed,
Z-up, centimetre frame:

```
p_ue = 100 * (x, -y, z)          R_ue = M R M,  M = diag(1, -1, 1)
```

The conversions live in [`physscene/geometry.py`](../physscene/geometry.py) and are unit-tested.
