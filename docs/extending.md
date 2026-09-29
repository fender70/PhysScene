# Extending PhysScene

## New event types

Register a template that returns a `Layout`. Mark the main body `name="object"`
and give it `stencil=1`:

```python
from physscene.events import register_event, EventContext, _make_object, _make_prop
from physscene.spec import Layout

@register_event("topple")
def topple(ctx: EventContext) -> Layout:
    """Roll the object into a tall prop so the prop tips over."""
    ...
    return Layout(event="topple", level=ctx.level.name, surface=ctx.surface.name,
                  bodies=[obj, prop], meta={"direction": ...})
```

Then add a branch to `physscene.checks.check_physics` that verifies the event
(for example, the prop's up-axis tilts more than 60°). Return `hint="faster"` or
`hint="slower"` on failure so `calibrate_speed` can fix the push. Import your
module before planning, or add it to `physscene/events.py`.

Template parameters come from `events.<name>` in the experiment config and
are merged over `events.DEFAULTS`.

## New interventions

The grid is `events × scenes × objects × appearances × views`, and each cell
can have futures. Other factors can be added as extra keys:

* **Lighting**: add a `lighting` factor to jobs, then in
  `unreal/physscene_ue/sequence.py` spawn or modify lights (or switch sublevels).
* **Physical parameter sweeps** (instead of random futures): use `futures.perturb`
  with `range:` and `count: N`, or call `planner.perturb` directly with
  hand-built specs.

## New physics backends

Implement `physscene.physics.base.PhysicsBackend.simulate` (for example with
PyBullet, Genesis or PhysX) and register it in `physscene/physics/__init__.py`.
Poses must be those of the collision primitive in the right-handed frame.

## New renderers

A renderer consumes `jobs/<job_id>.json` and writes
`renders/<job_id>/{rgb,labels,depth}/` plus a `DONE` marker. See
`physscene/preview.py` for a minimal example. Blender or Isaac Sim could be
added the same way.
