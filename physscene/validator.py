"""Independent physical-validity validator.

This module judges a trajectory using only saved state (``states.json``: layout,
physical parameters, per-frame poses) and basic mechanics. It deliberately
shares **no code** with the planner's event checks or the violation
generator. Positions are differentiated here, and the simulator's own
velocities and contact flags are ignored, so it provides an independent check
of every ground-truth label.

Laws checked per dynamic body:

* **persistence**: a body never disappears;
* **ballistic motion**: with no support or other body nearby, the acceleration
  equals gravity, ``a = (0, 0, -g)``;
* **Coulomb bound**: touching only horizontal supports,
  ``|a_h| <= mu * (a_z + g)``, so friction cannot push harder than the normal force;
* **angular momentum**: a sphere (or a cylinder lying on its side) rolling on
  one horizontal plane conserves angular momentum about the contact point, so
  it cannot stop, reverse or speed up on its own;
* **energy**: total mechanical energy of all bodies never rises;
* **non-penetration**: no body sinks into a support, and no two bodies overlap.

Tolerances absorb finite-difference and soft-contact error. The benchmark
pipeline reports the validator's agreement with the labels as a
decision-gate check.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

G = 9.81

DEFAULT_TOL: dict[str, float] = {
    "ballistic_abs": 1.5,  # m/s^2
    "coulomb_abs": 1.5,  # m/s^2
    "coulomb_rel": 0.3,
    "angmom_rel": 0.25,  # relative change of contact angular momentum per frame
    "energy_rel": 0.25,  # of the current kinetic energy
    "energy_abs": 0.0015,  # J
    "energy_lag": 3,  # frames
    "penetration": 0.012,  # m
    "near": 0.006,  # contact proximity (m)
    "edge_band": 0.01,  # m; centre this close to a surface edge = edge contact
}


# --------------------------------------------------------------------------- #
# Minimal geometry (re-implemented on purpose; no planner imports)
# --------------------------------------------------------------------------- #
def _qmat(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _half_extents(shape: dict) -> np.ndarray:
    t = shape["type"]
    if t == "sphere":
        return np.full(3, shape["radius"])
    if t == "box":
        return np.asarray(shape["half_extents"], float)
    extra = shape["radius"] if t == "capsule" else 0.0
    return np.array([shape["radius"], shape["radius"], shape["half_length"] + extra])


def _support_height(shape: dict, q) -> float:
    """Distance from the body centre to its lowest point."""
    t = shape["type"]
    if t == "sphere":
        return shape["radius"]
    down = _qmat(q).T @ np.array([0.0, 0.0, -1.0])
    if t == "box":
        return float(np.sum(np.abs(down) * np.asarray(shape["half_extents"])))
    axial = abs(down[2]) * shape["half_length"]
    if t == "capsule":
        return axial + shape["radius"]
    return axial + math.sqrt(max(0.0, 1 - down[2] ** 2)) * shape["radius"]


def _inertia_body(shape: dict, m: float) -> np.ndarray:
    t = shape["type"]
    if t == "sphere":
        return np.full(3, 0.4 * m * shape["radius"] ** 2)
    if t == "box":
        a, b, c = (2 * np.asarray(shape["half_extents"])) ** 2
        return m / 12 * np.array([b + c, a + c, a + b])
    r, L = shape["radius"], 2 * shape["half_length"]
    ixx = m * (3 * r * r + L * L) / 12
    return np.array([ixx, ixx, 0.5 * m * r * r])


def _combine(a: float, b: float, mode: str) -> float:
    if mode == "min":
        return min(a, b)
    if mode == "multiply":
        return a * b
    if mode == "max":
        return max(a, b)
    return 0.5 * (a + b)


def _inscribed(shape: dict) -> float:
    return float(min(_half_extents(shape)))


def _bounding(shape: dict) -> float:
    return shape["radius"] if shape["type"] == "sphere" else float(np.linalg.norm(_half_extents(shape)))


@dataclass
class SupportRect:
    center: np.ndarray
    half: np.ndarray
    yaw: float
    top: float

    def local(self, p) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        d = np.asarray(p[:2]) - self.center
        return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])

    def inside(self, p, margin=0.0) -> bool:
        lx, ly = self.local(p)
        return abs(lx) <= self.half[0] + margin and abs(ly) <= self.half[1] + margin

    def edge_distance(self, p) -> float:
        lx, ly = np.abs(self.local(p))
        return float(min(self.half[0] - lx, self.half[1] - ly))


@dataclass
class Finding:
    law: str
    body: str
    frame: int
    detail: str


@dataclass
class Verdict:
    valid: bool
    findings: list[Finding] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "summary": self.summary,
            "findings": [f.__dict__ for f in self.findings[:20]],
        }


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #
def validate_states(
    states: dict[str, Any],
    shapes: dict[str, dict],
    supports: list[dict],
    floor_z: float,
    tol: dict[str, float] | None = None,
) -> Verdict:
    """``states`` = {"layout", "params", "trajectory"} as saved in ``states.json``.
    ``shapes`` maps body name to its shape dict, and ``supports`` lists
    horizontal surfaces ``{center, size, yaw_deg}`` (``center`` is the top-face centre)."""
    tol = {**DEFAULT_TOL, **(tol or {})}
    tr = states["trajectory"]
    params = states["params"]
    fps = tr["fps"]
    dt = 1.0 / fps
    g = G  # judge against Earth gravity, not the (possibly broken) value the simulator was given
    names = tr["names"]
    bodies = {b["name"]: b for b in states["layout"]["bodies"]}
    rects = [
        SupportRect(np.asarray(s["center"][:2], float), np.asarray(s["size"], float) / 2, math.radians(s.get("yaw_deg", 0.0)), float(s["center"][2]))
        for s in supports
    ]
    T = len(tr["positions"][names[0]])
    findings: list[Finding] = []
    vis = {n: tr.get("visible", {}).get(n, [True] * T) for n in names}

    # persistence
    for n in names:
        missing = [t for t in range(T) if not vis[n][t]]
        if missing:
            findings.append(Finding("persistence", n, missing[0], f"body absent in {len(missing)} frames"))

    dyn = [n for n in names if bodies[n].get("dynamic", True)]
    P = {n: np.asarray(tr["positions"][n], float) for n in names}
    Q = {n: np.asarray(tr["quats"][n], float) for n in names}

    def near_other(n, t) -> bool:
        for m in names:
            if m == n:
                continue
            d = np.linalg.norm(P[n][t] - P[m][t])
            if d < _bounding(shapes[n]) + _bounding(shapes[m]) + tol["near"]:
                return True
        return False

    def support_state(n, t, v) -> str:
        """'free' | 'support' (flat horizontal contact) | 'complex' (edge,
        impact within the frame interval, or unknown). The contact margin grows
        with speed, since a bounce can happen between two sampled frames."""
        p, h = P[n][t], _support_height(shapes[n], Q[n][t])
        low = p[2] - h
        r = _bounding(shapes[n])
        near = tol["near"]
        reach = near + (abs(v[2]) + 0.5 * g * dt) * dt * 1.5
        for rc in rects:
            if rc.inside(p, margin=r):
                gap = low - rc.top
                if -0.2 < gap <= reach:
                    # A body rests flat on the top while its centre is over the
                    # surface; it starts pivoting on the edge once the centre passes it.
                    band = tol["edge_band"] + float(np.hypot(v[0], v[1])) * dt * 1.5
                    if rc.edge_distance(p) < band:
                        return "complex"
                    return "support" if gap <= near and abs(v[2]) < 0.3 else "complex"
        gap = low - floor_z
        if gap <= reach:
            return "support" if gap <= near and abs(v[2]) < 0.3 else "complex"
        return "free"

    def fd_vel(n, t) -> np.ndarray:
        a, b = max(0, t - 1), min(T - 1, t + 1)
        return (P[n][b] - P[n][a]) / ((b - a) * dt) if b > a else np.zeros(3)

    def contact_mu(n) -> float:
        pp = params["bodies"].get(n, {})
        mode = pp.get("friction_combine", "average")
        mus = [
            _combine(float(pp.get("friction", 0.6)), float(sup.get("friction", 0.6)), mode)
            for sup in (params.get("surface", {}), params.get("floor", {}))
        ]
        return max(mus)

    def omega(n, t) -> np.ndarray:
        """World angular velocity over [t, t+1] from consecutive orientations."""
        dq = _qmat(Q[n][t + 1]) @ _qmat(Q[n][t]).T
        ang = math.acos(max(-1.0, min(1.0, (np.trace(dq) - 1) / 2)))
        if ang < 1e-9:
            return np.zeros(3)
        axis = np.array([dq[2, 1] - dq[1, 2], dq[0, 2] - dq[2, 0], dq[1, 0] - dq[0, 1]]) / (2 * math.sin(ang))
        return axis * ang / dt

    masses = {n: float(params["bodies"].get(n, {}).get("mass", 1.0)) for n in dyn}
    inertias = {n: _inertia_body(shapes[n], masses[n]) for n in dyn}

    def kinetic(n, t) -> float:
        v = (P[n][t + 1] - P[n][t]) / dt
        wl = _qmat(Q[n][t]).T @ omega(n, t)
        return 0.5 * masses[n] * float(v @ v) + 0.5 * float(wl @ (inertias[n] * wl))

    def ang_momentum_contact(n, t):
        """Angular momentum about the (instantaneous) contact point, or None if
        the law does not apply to this shape/pose. Conserved for a sphere, and
        along the axis for a cylinder lying on its side, when the only contact is
        one horizontal plane (rolling resistance aside)."""
        shape = shapes[n]
        v = (P[n][t + 1] - P[n][t]) / dt
        w = omega(n, t)
        R = _qmat(Q[n][t])
        Iw = R @ (inertias[n] * (R.T @ w))
        h = _support_height(shape, Q[n][t])
        L = Iw + masses[n] * np.cross(np.array([0.0, 0.0, h]), v)
        if shape["type"] == "sphere":
            return L, masses[n] * h * 0.06 + 1e-6
        if shape["type"] in ("cylinder", "capsule"):
            axis = R[:, 2]
            if abs(axis[2]) < 0.15:
                return np.array([float(L @ axis)]), masses[n] * h * 0.06 + 1e-6
        return None

    for n in dyn:
        shape = shapes[n]
        ok = np.array(vis[n], bool)
        mu = contact_mu(n)

        def state(k) -> str:
            return support_state(n, k, fd_vel(n, k))

        for t in range(1, T - 1):
            if not (ok[t - 1] and ok[t] and ok[t + 1]):
                continue
            a = (P[n][t + 1] - 2 * P[n][t] + P[n][t - 1]) / dt**2
            window = [state(k) for k in (t - 1, t, t + 1)]
            other = any(near_other(n, k) for k in (t - 1, t, t + 1))
            if other or "complex" in window:
                continue
            if all(s == "free" for s in window):
                err = np.linalg.norm(a - np.array([0, 0, -g]))
                if err > tol["ballistic_abs"]:
                    findings.append(Finding("ballistic", n, t, f"|a - g| = {err:.2f} m/s^2 in free flight"))
            elif all(s == "support" for s in window):
                ah = float(np.hypot(a[0], a[1]))
                bound = mu * max(0.0, a[2] + g) * (1 + tol["coulomb_rel"]) + tol["coulomb_abs"]
                if ah > bound:
                    findings.append(Finding("coulomb", n, t, f"|a_h| = {ah:.2f} > {bound:.2f} m/s^2 on a support"))
                if t + 1 < T - 1 and ok[t + 1] and ok[t - 1]:
                    l0 = ang_momentum_contact(n, t - 1)
                    l1 = ang_momentum_contact(n, t)
                    if l0 is not None and l1 is not None and len(l0[0]) == len(l1[0]):
                        dl = float(np.linalg.norm(l1[0] - l0[0]))
                        lim = tol["angmom_rel"] * max(np.linalg.norm(l0[0]), np.linalg.norm(l1[0])) + l0[1]
                        if dl > lim:
                            findings.append(Finding("angular_momentum", n, t, f"|dL| = {dl:.2e} > {lim:.2e} while rolling"))

        # penetration into supports / floor
        for t in range(T):
            if not ok[t]:
                continue
            p, h = P[n][t], _support_height(shape, Q[n][t])
            low = p[2] - h
            if low < floor_z - tol["penetration"]:
                findings.append(Finding("penetration", n, t, f"{floor_z - low:.3f} m below the floor"))
                break
            hit = False
            for rc in rects:
                if rc.inside(p) and rc.top - 0.5 < low < rc.top - tol["penetration"] and p[2] < rc.top + h:
                    findings.append(Finding("penetration", n, t, f"{rc.top - low:.3f} m into a support"))
                    hit = True
                    break
            if hit:
                break

    # total mechanical energy of all dynamic bodies must not rise
    # (compared with the minimum up to `lag` frames earlier, so the brief dip
    # while energy sits in a soft contact is not mistaken for a gain)
    lag = int(tol["energy_lag"])
    E, KE = [], []
    for t in range(T - 1):
        if not all(vis[n][t] and vis[n][t + 1] for n in dyn):
            E.append(np.nan)
            KE.append(np.nan)
            continue
        ke = sum(kinetic(n, t) for n in dyn)
        pe = sum(masses[n] * g * 0.5 * (P[n][t][2] + P[n][t + 1][2]) for n in dyn)
        E.append(ke + pe)
        KE.append(ke)
    # Intervals containing an impact under-estimate kinetic energy (the velocity
    # reverses inside the interval), so they must not set the baseline.
    def abrupt(n, t) -> bool:
        """Large change of linear or rim velocity between neighbouring intervals."""
        if t < 1 or t + 2 >= T:
            return True
        r = _bounding(shapes[n])
        v_prev = (P[n][t] - P[n][t - 1]) / dt
        v_next = (P[n][t + 2] - P[n][t + 1]) / dt
        w_prev, w_next = omega(n, t - 1), omega(n, t + 1)
        return bool(np.linalg.norm(v_next - v_prev) > 0.25 or np.linalg.norm(w_next - w_prev) * r > 0.25)

    impact = [
        any(
            support_state(n, k, fd_vel(n, k)) == "complex" or near_other(n, k)
            for n in dyn
            for k in (t, t + 1)
        )
        or any(abrupt(n, t) for n in dyn if vis[n][t] and (t + 2 >= T or vis[n][t + 2]))
        for t in range(len(E))
    ]
    for t in range(lag, len(E)):
        prev = [e for k, e in enumerate(E[: t - lag + 1]) if np.isfinite(e) and not impact[k]]
        if not prev or not np.isfinite(E[t]):
            continue
        rise = E[t] - min(prev)
        etol = tol["energy_rel"] * KE[t] + tol["energy_abs"]
        if rise > etol:
            findings.append(Finding("energy", "all", t, f"energy rose by {rise:.4f} J (tol {etol:.4f})"))
            break

    # body-body overlap (inscribed spheres must not intersect)
    for i, n in enumerate(names):
        for m_ in names[i + 1 :]:
            lim = _inscribed(shapes[n]) + _inscribed(shapes[m_]) - tol["penetration"]
            for t in range(T):
                if vis[n][t] and vis[m_][t] and np.linalg.norm(P[n][t] - P[m_][t]) < lim:
                    findings.append(Finding("penetration", f"{n}+{m_}", t, "bodies overlap"))
                    break

    summary: dict[str, int] = {}
    for f in findings:
        summary[f.law] = summary.get(f.law, 0) + 1
    return Verdict(valid=not findings, findings=findings, summary=summary)


def validate_states_file(path, tol=None) -> Verdict:
    """Validate an exported ``states.json`` (which carries shapes and world geometry)."""
    import json
    from pathlib import Path

    states = json.loads(Path(path).read_text())
    w = states["world"]
    return validate_states(states, states["shapes"], w["supports"], w["floor_z"], tol)
