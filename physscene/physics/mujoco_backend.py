"""MuJoCo rigid-body backend.

MuJoCo is deterministic on a given machine and build, installs as a wheel on
every major platform and supports rolling and torsional friction (condim 6).
That makes it a good reference simulator for datasets that must be exactly
reproducible. The trajectory is then baked into Unreal as keyframes, so every
viewpoint and appearance intervention shows **the same** physical event.
"""
from __future__ import annotations

import math
from xml.sax.saxutils import quoteattr

import numpy as np

from ..catalog import Catalog, Level, Shape
from ..geometry import quat_from_yaw, quat_to_matrix
from ..spec import Layout, PhysicalParams, Trajectory
from .base import PhysicsBackend


def restitution_to_dampratio(e: float) -> float:
    """Damping ratio of the contact spring that gives a coefficient of
    restitution ``e`` for a linear spring-damper."""
    e = float(np.clip(e, 1e-3, 0.99))
    ln = math.log(e)
    return -ln / math.sqrt(math.pi**2 + ln**2)


def _geom_attrs(shape: Shape) -> str:
    if shape.type == "sphere":
        return f'type="sphere" size="{shape.radius}"'
    if shape.type == "box":
        hx, hy, hz = shape.half_extents
        return f'type="box" size="{hx} {hy} {hz}"'
    return f'type="{shape.type}" size="{shape.radius} {shape.half_length}"'


def _fmt(v) -> str:
    return " ".join(f"{float(x):.9g}" for x in v)


COMBINE_ORDER = ["average", "min", "multiply", "max"]


def combine(a: float, b: float, mode_a: str = "average", mode_b: str = "average") -> float:
    """Unreal-style property combination: the higher-priority mode of the two wins."""
    mode = max(mode_a, mode_b, key=COMBINE_ORDER.index)
    if mode == "min":
        return min(a, b)
    if mode == "multiply":
        return a * b
    if mode == "max":
        return max(a, b)
    return 0.5 * (a + b)


def _solref(restitution: float, dt: float) -> str:
    tc = max(0.004, 2.5 * dt)
    return f"{tc:.6g} {restitution_to_dampratio(restitution):.6g}"


def build_mjcf(layout: Layout, params: PhysicalParams, level: Level, catalog: Catalog, timestep: float) -> str:
    """Build the MJCF model.

    Every contact is declared as an explicit ``<pair>`` so we control how the
    material properties combine. We use the **average** of both geoms'
    friction and restitution by default, which is Unreal Chaos's default combine mode, and
    honour per-asset ``friction_combine`` / ``restitution_combine`` like
    Unreal physical materials do. (MuJoCo's default would take the max
    friction, so a low-friction toy car on a wooden table could never
    slide.) Rolling and torsional friction come from the dynamic body."""
    surf = level.surfaces[layout.surface]
    sshape, scenter, squat = surf.box_shape()
    nocol = 'contype="0" conaffinity="0"'
    lines = [
        '<mujoco model="physscene">',
        f'  <option timestep="{timestep}" gravity="0 0 {-params.gravity}" integrator="implicitfast" '
        'cone="elliptic" impratio="10"/>',
        '  <default><geom condim="6" solimp="0.95 0.99 0.001"/></default>',
        "  <worldbody>",
        f'    <geom name="floor" type="plane" size="0 0 1" pos="0 0 {level.floor_z}" {nocol}/>',
        f'    <geom name="surface" {_geom_attrs(sshape)} pos="{_fmt(scenter)}" quat="{_fmt(squat)}" {nocol}/>',
    ]
    # (name, friction, restitution) for everything a dynamic body can touch
    statics = [("floor", params.floor["friction"], params.floor["restitution"], "average", "average"),
               ("surface", params.surface["friction"], params.surface["restitution"], "average", "average")]
    for c in level.colliders:
        name = "static_" + c.name
        lines.append(
            f'    <geom name={quoteattr(name)} {_geom_attrs(c.shape)} pos="{_fmt(c.center)}" '
            f'quat="{_fmt(quat_from_yaw(math.radians(c.yaw_deg)))}" {nocol}/>'
        )
        statics.append((name, 0.5, 0.2, "average", "average"))
    dynamic = []
    for b in layout.bodies:
        a = catalog.get(b.asset)
        p = params.bodies[b.name]
        if b.dynamic:
            lines += [
                f'    <body name="{b.name}" pos="{_fmt(b.position)}" quat="{_fmt(b.quat)}">',
                f'      <joint name="{b.name}_free" type="free" damping="{p["linear_damping"]:.6g}"/>',
                f'      <geom name="{b.name}" {_geom_attrs(a.shape)} mass="{p["mass"]:.6g}" {nocol}/>',
                "    </body>",
            ]
            dynamic.append(b.name)
        else:
            lines.append(f'    <geom name="{b.name}" {_geom_attrs(a.shape)} pos="{_fmt(b.position)}" quat="{_fmt(b.quat)}" {nocol}/>')
            statics.append((b.name, p["friction"], p["restitution"], p.get("friction_combine", "average"), p.get("restitution_combine", "average")))
    lines.append("  </worldbody>")
    lines.append("  <contact>")

    def pair(a: str, b: str, pa: tuple, pb: tuple, roll: float, tors: float):
        f = combine(pa[0], pb[0], pa[2], pb[2])
        e = combine(pa[1], pb[1], pa[3], pb[3])
        lines.append(
            f'    <pair geom1={quoteattr(a)} geom2={quoteattr(b)} condim="6" '
            f'friction="{f:.6g} {f:.6g} {tors:.6g} {roll:.6g} {roll:.6g}" solref="{_solref(e, timestep)}"/>'
        )

    def props(n: str) -> tuple:
        p = params.bodies[n]
        return (p["friction"], p["restitution"], p.get("friction_combine", "average"), p.get("restitution_combine", "average"))

    for i, n in enumerate(dynamic):
        p = params.bodies[n]
        for st in statics:
            pair(n, st[0], props(n), st[1:], p["rolling_friction"], p["torsional_friction"])
        for m in dynamic[i + 1 :]:
            q = params.bodies[m]
            pair(n, m, props(n), props(m),
                 max(p["rolling_friction"], q["rolling_friction"]), max(p["torsional_friction"], q["torsional_friction"]))
    lines += ["  </contact>", "</mujoco>"]
    return "\n".join(lines)


class MujocoBackend(PhysicsBackend):
    name = "mujoco"

    def simulate(
        self,
        layout: Layout,
        params: PhysicalParams,
        level: Level,
        catalog: Catalog,
        fps: int,
        num_frames: int,
        timestep: float = 0.001,
        pre_roll: float = 0.0,
    ) -> Trajectory:
        import mujoco

        xml = build_mjcf(layout, params, level, catalog, timestep)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)

        dyn = [b for b in layout.bodies if b.dynamic]
        for b in dyn:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{b.name}_free")
            qa, va = model.jnt_qposadr[jid], model.jnt_dofadr[jid]
            data.qpos[qa : qa + 3] = b.position
            data.qpos[qa + 3 : qa + 7] = b.quat
            r = quat_to_matrix(b.quat)
            data.qvel[va : va + 3] = b.lin_vel
            data.qvel[va + 3 : va + 6] = r.T @ np.asarray(b.ang_vel, float)  # free-joint omega is body-local
        # The joint's damping attribute covers all six dofs; give the rotational
        # dofs their own angular damping instead.
        for b in dyn:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{b.name}_free")
            va = model.jnt_dofadr[jid]
            model.dof_damping[va + 3 : va + 6] = params.bodies[b.name]["angular_damping"]
        mujoco.mj_forward(model, data)

        geom_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(model.ngeom)]
        substeps = max(1, int(round(1.0 / (fps * timestep))))
        names = [b.name for b in layout.bodies]
        body_ids = {
            b.name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b.name) for b in dyn
        }
        static_pose = {b.name: (list(b.position), list(b.quat)) for b in layout.bodies if not b.dynamic}

        for _ in range(int(round(pre_roll / timestep))):
            mujoco.mj_step(model, data)

        positions = {n: [] for n in names}
        quats = {n: [] for n in names}
        lin_vel = {n: [] for n in names}
        contacts: list[list[list[str]]] = []

        def record(frame_contacts: set[tuple[str, str]]):
            for n in names:
                if n in body_ids:
                    bid = body_ids[n]
                    positions[n].append(data.xpos[bid].tolist())
                    quats[n].append(data.xquat[bid].tolist())
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_free")
                    va = model.jnt_dofadr[jid]
                    lin_vel[n].append(data.qvel[va : va + 3].tolist())
                else:
                    positions[n].append(static_pose[n][0])
                    quats[n].append(static_pose[n][1])
                    lin_vel[n].append([0.0, 0.0, 0.0])
            contacts.append(sorted([list(p) for p in frame_contacts]))

        def current_contacts() -> set[tuple[str, str]]:
            out = set()
            for i in range(data.ncon):
                c = data.contact[i]
                a, b = geom_names[c.geom1], geom_names[c.geom2]
                out.add(tuple(sorted((a, b))))
            return out

        record(current_contacts())
        for _ in range(num_frames - 1):
            acc: set[tuple[str, str]] = set()
            for _ in range(substeps):
                mujoco.mj_step(model, data)
                acc |= current_contacts()
            record(acc)

        return Trajectory(
            fps=fps, names=names, positions=positions, quats=quats, contacts=contacts, lin_vel=lin_vel
        )
