# Rendering with Unreal Engine 5

PhysScene's Unreal driver lives in [`unreal/`](../unreal). It runs inside the
editor's Python interpreter and needs no extra Python packages. It was written
against the UE 5.3–5.7 Python API. Renamed APIs are handled in
[`ue_compat.py`](../unreal/physscene_ue/ue_compat.py).

## 1. Project setup (once)

1. Create or open a UE 5.x project. The *Blank* template is enough for the starter experiment.
2. **Edit → Plugins**: enable **Python Editor Script Plugin**, **Movie Render
   Queue** and **Movie Render Queue Additional Render Passes**. Restart.
3. **Project Settings → Rendering → Postprocessing → Custom Depth-Stencil
   Pass = Enabled with Stencil**. The driver also sets `r.CustomDepth 3` per
   render job, but the project setting avoids a shader recompile at render time.
4. Optional: **Project Settings → Python → Additional Paths**, add
   `<PhysScene>/unreal`, so `import physscene_ue` works in the editor console.

## 2. Plan, then render

```bash
pip install -e .                         # on the machine that runs Unreal
physscene plan configs/starter.yaml -o out/starter/plan
physscene render out/starter/plan --renderer unreal \
    --uproject "C:/Projects/MyProj/MyProj.uproject" \
    --editor "C:/Program Files/Epic Games/UE_5.7/Engine/Binaries/Win64/UnrealEditor.exe"
physscene export out/starter/plan -o out/starter/dataset
physscene validate out/starter/dataset
```

`--dry-run` prints the exact editor command and environment without running
it. The render step:

1. waits for the editor to finish loading;
2. groups jobs by level, and loads each level in turn;
3. builds one Level Sequence per job under `/Game/PhysSceneGenerated/…`,
   containing spawnable mesh actors with baked transform keys, optional
   procedural surfaces (`build_geometry: true`), and a CineCamera with
   filmback and focal length matching the planned FOV;
4. queues up to three MRQ jobs per PhysScene job:
   `rgb` (PNG, anti-aliased), `labels` (stencil-bit PNG, no AA) and `depth`
   (EXR, metres);
5. writes `<render_root>/<job_id>/DONE` when all passes succeed. Re-running the
   command skips finished jobs, so interrupted batches resume;
6. quits the editor when everything is done.

You can also run [`unreal/run_jobs.py`](../unreal/run_jobs.py) from an open
editor (**Tools → Execute Python Script**) after setting `PHYSSCENE_PLAN` in
the environment.

## 3. Using your own assets and levels

* **Assets**: copy [`configs/catalog_cronos_template.yaml`](../configs/catalog_cronos_template.yaml),
  set `ue_mesh` and the appearance materials to your assets, and fit the collision
  primitive (`shape`) and `pivot_offset` to the mesh. Materials can be asset
  paths or `{base, vector_params, scalar_params, texture_params}` dicts. The
  driver creates the material instances.
* **Levels**: tag the tables and counters in your level with `PhysSceneSurface`
  (optional `PhysScenePrompt=kitchen counter`, `PhysSceneFriction=0.4`,
  `PhysSceneFloorLevel`), obstacles with `PhysSceneCollider`, a camera-limit
  volume with `PhysSceneCameraBounds` and clutter to hide with `PhysSceneHide`.
  Then, in the editor Python console:

  ```python
  import physscene_ue.level_export as le
  le.export_level("C:/PhysScene/configs/levels/kitchen.yaml", name="kitchen")
  ```

  Add the manifest to your experiment config's `levels:` list.

## 4. Known limitations

* The planner does not know about level geometry that is not declared as a
  surface or collider. For example, a cupboard between the camera and the
  table can occlude the event. Declare big occluders as `colliders`, or
  restrict the camera with `bounds_min/bounds_max`.
* Collision proxies are primitives (sphere, box, cylinder, capsule). For
  complex meshes, pick the primitive that best matches the contact behaviour.
* The driver is tested against a mocked `unreal` module in CI. Real-engine
  behaviour can vary between engine versions. Please open an issue with the
  editor log (`Saved/Logs`) if something breaks.
