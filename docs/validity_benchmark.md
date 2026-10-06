# Physical-validity benchmarks

This mode builds datasets for testing whether a **metric or model tracks
physical validity**. Each reference clip gets a set of candidate futures that
share its conditioning prefix exactly:

```
frames 0 ────────── P │ P+1 ───────────────────── T-1
reference   ██████████│████████████████████████████   the simulated future
valid f01   ██████████│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓   branch at P, resampled hidden physics
invalid     ██████████│░░░░░░░░░░░░░░░░░░░░░░░░░░░░   branch at P, controlled law violation
            identical │ diverges
```

Ground-truth labels come from **how each candidate was built**. They are then
**re-derived independently** from the saved simulator state. No metric result
ever defines ground truth.

## Candidate types

| kind | construction | severities |
|---|---|---|
| `valid` (`alternatives/fNN`) | re-simulate from the exact state at frame P with resampled friction, restitution, rolling friction, mass, surface friction and collider mass/friction; no state edits | the perturbation ranges in the config |
| `teleport` | object position jumps sideways, then continues; the jump never moves it off its support (so it cannot also hover) | low / mid / high = 5 / 15 / 30 cm |
| `speed_jump` | velocity multiplied at P (dynamic re-simulation) | ×1.5 / ×2.5 / ×4 |
| `gravity` | re-simulate with wrong gravity (fall events only) | ×0.5 / ×0.2 / ×0 (weightless) |
| `freeze` | object stops dead | n/a |
| `time_reversal` | motion plays backwards after P | n/a |
| `vanish` | object disappears (low/mid: reappears after 8 or 16 frames; high: never returns) | low / mid / high |
| `penetration` | contact with the next support or collider disabled (falls through the table, passes through the cup) | n/a |

A candidate is kept only if it is **visibly different** from the reference:

* in 3D (`min_divergence`, max position deviation of any body), and
* **in every camera view**: at least 3 frames after the prefix where the object
  appears or disappears relative to the reference, or is `min_pixel_shift`
  (6 px at 1280 width) away from its reference position. A `vanish` while the
  object is already behind an occluder is therefore rejected, and vanish onsets are
  scheduled when the object is visible in every view;
* with the object **in frame** for at least `min_in_frame_frac` (50%) of the
  frames after the prefix (except `vanish` and occlusion events).

With
`require_validator: true` (the default) it must also be **independently
confirmed**: violations must be flagged and valid alternatives must pass.
Rejections are counted in `plan.json` and in the gate report.

## Independent validator

[`physscene/validator.py`](../physscene/validator.py) reads only
`states.json`: poses, parameters, shapes and support geometry. It computes
velocities by finite differences and ignores the simulator's own velocities
and contact flags. It shares no code with the generator. It checks:

* **persistence**: bodies never disappear;
* **ballistic motion**: in free flight, `a = (0, 0, −9.81)`;
* **Coulomb bound**: on a horizontal support, `|a_h| ≤ μ (a_z + g)`;
* **rolling angular momentum**: a sphere, or a cylinder lying on its side,
  conserves angular momentum about the contact point, so it cannot stop,
  reverse or speed up on its own;
* **energy**: total mechanical energy never rises (impact intervals excluded
  from the baseline);
* **non-penetration**: no sinking into supports or overlapping bodies.

**Held-out accuracy.** The tolerances were tuned on the starter shapes. On
held-out assets and a held-out level (the CRONOS-style catalog in the kitchen
template: tennis ball, soccer ball, can, bottle and toy truck; seed 4242), with
filtering disabled, the validator accepted **88/90** valid futures and
**15/15** references, and flagged **189/191** violations. The two misses were
`freeze` on slow objects, where stopping is hard to tell apart from ordinary
deceleration. Reproduce with `require_validator: false` and the counts in
`plan.json`.

References must pass the validator too. On our assets this caught a MuJoCo
artefact, a cylinder gaining energy while rolling over a box edge; such layouts
are rejected and re-sampled.

## Outputs

```
dataset/<event>/<scene>/<object>/<appearance>/<view>/          reference
                                               .../alternatives/f01/  valid candidate
                                               .../violations/teleport_high/  invalid candidate
dataset/benchmark/pairs.jsonl          one line per (reference, candidate): label, type, severity,
                                       prefix_frames, onset_frame, divergence, validator agreement
dataset/benchmark/summary.json         counts per candidate type + dataset hash
dataset/benchmark/frozen_manifest.json SHA-256 of every video / metadata / state file
dataset/benchmark/GATE_REPORT.md       decision-gate report (see below)
```

## Decision-gate report

```bash
physscene gate PLAN_DIR DATASET_DIR
```

| criterion | how it is tested |
|---|---|
| G1 reproducible initial conditions and seeds | re-plan sampled groups from config and seed (layouts, trajectories and candidates bit-identical); re-simulate references from the saved state alone (0 error) |
| G2 programmatic control over physical properties | set each hidden parameter to 0.5× and 2× at the prefix; the value reaches the simulator model and the trajectory responds |
| G3 per-frame state sufficient for independent checks | validator on every exported `states.json` vs labels (confusion matrix per violation type) |
| G4 reproducible counterfactual visual transformations | identical state across every view and appearance; bit-identical state and pixels over the matched prefix; masks agree with projected state; a job re-renders bit-identically |
| G5 stable batch rendering without manual adjustment | every job rendered with a matching content fingerprint and exported; all groups planned; config unchanged since planning |
| G6 complete provenance | every sample carries factors, candidate, physics, render settings, code commit, config hash and job fingerprint; frozen manifest hashes verified |
| Q benchmark item quality | every candidate differs from its reference in rendered pixels after the prefix, in every view; the object stays in view for at least half of the frames after the prefix |

G3 also reports the validator's disagreement **before** filtering. The
confusion matrix over the exported dataset is 100% by construction when
`require_validator` is on, so the pre-filter counts are the real error rate.

It also reports the validation-before-scale stages: physics-only checks →
one-render smoke test → matched-prefix checks → full rendering → independent
state validation → frozen scoring manifest.

## Configuration

See [`configs/validity_pilot.yaml`](../configs/validity_pilot.yaml):

```yaml
benchmark:
  prefix_frames: 24
  require_validator: true
  valid:
    count: 2
    min_divergence: 0.02
    perturb: {friction: {rel: 0.4}, restitution: {rel: 0.4}, mass: {rel: 0.4}, collider.mass: {rel: 0.6}}
  invalid:
    min_divergence: 0.03
    types: {teleport: [low, high], speed_jump: [low, high], gravity: [mid, high], freeze: [mid],
            time_reversal: [mid], vanish: [low, high], penetration: [mid]}
    applicable_events: {gravity: [fall]}
```

The key event is forced to happen after the prefix: the planner raises the
lower bound of `checks.event_window` to frame P + 3.
