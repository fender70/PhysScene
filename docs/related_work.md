# Related work and the automation gap

This note surveys how researchers generate synthetic physics videos, with a
focus on Unreal Engine. It explains why PhysScene exists.

## The problem

Physics-aware video models (video generators, world models, VLMs) are
increasingly evaluated with **intervention-based** benchmarks. You hold a
physical event fixed and change one factor (viewpoint, scene, object
category, appearance), or you hold the initial state fixed and ask for
**several physically plausible futures**. Such benchmarks need:

1. photorealistic rendering (so conclusions transfer to real video);
2. exact control over initial conditions and physical parameters;
3. many controlled variants of the same event;
4. the simulator state, so others can regenerate, extend or re-render.

Unreal Engine covers (1) well. In practice, (2)–(4) are done by hand.

## Unreal Engine benchmarks

| Work | What it provides | Scene generation |
|---|---|---|
| **CRONOS** (Begiristain, Dünkel, Kortylewski, 2026) — [arXiv 2605.23699](https://arxiv.org/abs/2605.23699), [code](https://github.com/GenIntel/CRONOS-benchmark) | Counterfactual physical-consistency benchmark: events *fall / collision / occlusion*, interventions on scene (5), object (5), appearance (3/object) and viewpoint. Evaluation uses SAM3, CoTracker3, DINOv2, DisMo, SAM3D and a VLM judge. | Public release has evaluation code and data only. No Unreal scene-generation pipeline, simulator state or generation parameters. |
| **VideoPhysEdit** (2026) — [arXiv 2609.35134](https://arxiv.org/html/2609.35134) | 129 physical counterfactual *editing* tasks (velocity, mass, friction, restitution, insertion/removal) with ground truth. | Task-specific; not a general generator. |
| **SegGen** (2025) — [PMC12431427](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12431427/) | Automated UE5 pipeline for semantic segmentation datasets (procedural biomes, drone flights). | Automated, but static scenes and camera paths only; no physical events. |
| **UE world-model pretraining pipeline** (2026) — [arXiv 2609.03557](https://arxiv.org/abs/2609.03557) | Large-scale action-conditioned video; Stage I runs physics in PIE and records states, Stage II renders. | Automated for character locomotion; not object-physics events or interventions. |
| **StereoGenBench** (2026) — [arXiv 2605.23237](https://arxiv.org/pdf/2605.23237) | UE Python pipeline driving Movie Render Queue for multi-camera stereo sequences. | Automated camera rigs; no physics events. |
| **UnrealCV / UnrealZoo** — [UnrealCV](https://github.com/unrealcv/unrealcv), [UnrealZoo (ICCV'25)](https://mlanthology.org/iccv/2025/zhong2025iccv-unrealzoo/) | Python ↔ UE bridge; 100+ photoreal worlds for embodied AI. | Infrastructure for control and capture, not an event or intervention generator. |

## Non-Unreal generators

| Work | Engine | Notes |
|---|---|---|
| **Kubric** (CVPR'22) — [github](https://github.com/google-research/kubric) | PyBullet + Blender | The closest in spirit: scalable, seeded, rich annotations (masks, depth, flow). Physics is simulated first and then rendered. Semi-realistic Blender scenes; no counterfactual-intervention design. **PhysScene follows Kubric's simulate-then-render architecture with Unreal as the renderer.** |
| **Physion / Physion++** — [NeurIPS'23](https://proceedings.neurips.cc/paper_files/paper/2023/file/d3e8011c912e651ab2a76e7935a1e464-Paper-Datasets_and_Benchmarks.pdf) | ThreeDWorld (Unity + PhysX) | Scripted "controllers" generate physical-prediction stimuli; less photoreal than UE. |
| **CLEVRER** | Bullet + Blender | Collision videos with counterfactual questions; toy tabletop visuals. |
| **ThreeDWorld (TDW)** — [site](https://www.threedworld.org/) | Unity + PhysX | Python controller API; used by Physion. |
| **Infinigen + Omniverse Replicator** — [Isaac Sim docs](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/replicator_tutorials/tutorial_replicator_infinigen_sdg.html) | Blender / Omniverse | Procedural environments and SDG; general purpose, and no event templates for physics benchmarks. |

## The gap

As of September 2026 we found **no open-source tool that automatically
generates physics-event scenes in Unreal Engine**: no tool that places
objects, tunes pushes, picks cameras, verifies that the intended event
happens, renders masks and depth, and saves a reproducible simulator state.
Unreal-based physics benchmarks are assembled by hand. As a consequence:

* datasets cannot be extended (new objects, viewpoints, levels) by anyone but the authors;
* "alternative futures" from the *same* initial conditions cannot be regenerated;
* viewpoint and appearance interventions re-run live physics (Chaos is not
  guaranteed to be deterministic across runs), so "the same event" may differ
  slightly between variants.

## How PhysScene closes it

| Need | PhysScene |
|---|---|
| Per-scene manual placement | **Event templates** (fall / collision / occlusion; add your own) propose layouts from a one-time **level manifest** (surfaces), which can be exported from tagged actors. |
| Hand-tuned impulses | **Speed calibration**: simulate, check, and bisect the push until the event happens inside the time window. |
| "Does it look right?" | **Automatic checks**: physics (fell and landed, contact happened, passed the occluder) and camera (in frame, visible at frame 0, actually hidden then reappears). |
| Viewpoint choice | **Event-aware camera sampler** with stratified azimuths, auto distance, and occlusion-aware placement. |
| Same event across variants | Physics is **simulated once** (MuJoCo, deterministic) and **baked into keyframes**, so every view and appearance shows identical motion. |
| Multiple futures | **Futures sampler**: identical frame 0, resampled hidden physics (friction, restitution, mass, surface friction, initial-velocity noise), and the event re-verified. |
| No simulator state | Every sample ships `states.json` (layout, parameters, per-frame poses, Unreal keys) and `camera.json`. |
| Evaluation compatibility | Output layout and `metadata.json` match the CRONOS evaluation code. `cronos_config.json` provides the prompt dictionaries. |
