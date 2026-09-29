"""PhysScene Unreal Engine driver (runs inside the UE 5.x editor Python).

Pure Python + the ``unreal`` module, with no numpy, so it works with the stock
editor interpreter. All math (physics, camera placement, coordinate
conversion) happens offline in the ``physscene`` package. Each job file
already holds Unreal-space keyframes.

Entry points:
    run_jobs.py                     batch render a plan (used by ``physscene render --renderer unreal``)
    physscene_ue.level_export       write a level manifest from tagged actors
"""

__version__ = "0.1.0"
