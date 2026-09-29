"""Small, dependency-light geometry helpers.

Conventions
-----------
PhysScene works internally in a **right-handed, Z-up, metre** world frame
(the MuJoCo / PyBullet / ROS convention). Quaternions are ``(w, x, y, z)``.

Unreal Engine uses a **left-handed, Z-up, centimetre** frame. The mapping is a
mirror across the XZ plane plus a unit change::

    p_ue = 100 * (x, -y, z)

Rotations are converted through their matrices (``R_ue = M R M`` with
``M = diag(1, -1, 1)``) and expressed as Unreal ``FRotator`` angles
``(roll, pitch, yaw)`` in degrees, using the same decomposition as
``FMatrix::Rotator()``. See :func:`quat_to_ue_rotator`.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

M_TO_CM = 100.0
_MIRROR = np.diag([1.0, -1.0, 1.0])


def vec(v: Iterable[float]) -> np.ndarray:
    return np.asarray(list(v), dtype=float)


def normalize(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        raise ValueError("cannot normalize a zero-length vector")
    return a / n


# --------------------------------------------------------------------------- #
# Quaternions (w, x, y, z)
# --------------------------------------------------------------------------- #
def quat_identity() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_from_axis_angle(axis: Sequence[float], angle: float) -> np.ndarray:
    ax = normalize(axis)
    s = math.sin(angle / 2.0)
    return np.array([math.cos(angle / 2.0), *(ax * s)])


def quat_from_yaw(yaw: float) -> np.ndarray:
    return quat_from_axis_angle((0, 0, 1), yaw)


def quat_mul(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def rotate(q: Sequence[float], v: Sequence[float]) -> np.ndarray:
    return quat_to_matrix(q) @ np.asarray(v, dtype=float)


# --------------------------------------------------------------------------- #
# Unreal conversions
# --------------------------------------------------------------------------- #
def to_ue_location(p: Sequence[float]) -> list[float]:
    """Right-handed metres -> Unreal left-handed centimetres."""
    x, y, z = p
    return [x * M_TO_CM, -y * M_TO_CM, z * M_TO_CM]


def from_ue_location(p: Sequence[float]) -> np.ndarray:
    x, y, z = p
    return np.array([x / M_TO_CM, -y / M_TO_CM, z / M_TO_CM])


def ue_rotator_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Unreal ``FRotationMatrix`` (degrees) as a column-vector rotation matrix
    acting on Unreal (left-handed) coordinates. Columns are the images of the
    local X, Y, Z axes."""
    r, p, y = (math.radians(a) for a in (roll, pitch, yaw))
    sr, cr = math.sin(r), math.cos(r)
    sp, cp = math.sin(p), math.cos(p)
    sy, cy = math.sin(y), math.cos(y)
    x_axis = [cp * cy, cp * sy, sp]
    y_axis = [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp]
    z_axis = [-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp]
    return np.array([x_axis, y_axis, z_axis]).T


def ue_matrix_to_rotator(m: np.ndarray) -> list[float]:
    """Inverse of :func:`ue_rotator_to_matrix` (mirrors ``FMatrix::Rotator``)."""
    x_axis, y_axis, z_axis = m[:, 0], m[:, 1], m[:, 2]
    pitch = math.degrees(math.atan2(x_axis[2], math.hypot(x_axis[0], x_axis[1])))
    yaw = math.degrees(math.atan2(x_axis[1], x_axis[0]))
    roll = math.degrees(math.atan2(-y_axis[2], z_axis[2]))
    return [roll, pitch, yaw]


def quat_to_ue_rotator(q: Sequence[float]) -> list[float]:
    """Right-handed quaternion -> Unreal ``(roll, pitch, yaw)`` in degrees."""
    m_ue = _MIRROR @ quat_to_matrix(q) @ _MIRROR
    return ue_matrix_to_rotator(m_ue)


def ue_rotator_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    m_rh = _MIRROR @ ue_rotator_to_matrix(roll, pitch, yaw) @ _MIRROR
    return matrix_to_quat(m_rh)


def unwrap_degrees(seq: Sequence[Sequence[float]]) -> list[list[float]]:
    """Unwrap each Euler channel so consecutive keys never jump by >180 deg.

    Sequencer interpolates between keys; without unwrapping a 179 -> -179 deg
    step spins the object the long way round in motion-blur sub-samples."""
    arr = np.asarray(seq, dtype=float)
    if len(arr) == 0:
        return []
    return np.degrees(np.unwrap(np.radians(arr), axis=0)).tolist()


# --------------------------------------------------------------------------- #
# Cameras
# --------------------------------------------------------------------------- #
def look_at_matrix(eye: Sequence[float], target: Sequence[float], up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Camera-to-world rotation for a camera at ``eye`` looking at ``target``.

    Uses the Unreal-style camera frame expressed in the right-handed world:
    column 0 = forward, column 1 = left, column 2 = up."""
    fwd = normalize(np.asarray(target, float) - np.asarray(eye, float))
    left = np.cross(np.asarray(up, float), fwd)
    if np.linalg.norm(left) < 1e-9:  # looking straight up/down
        left = np.cross((1.0, 0.0, 0.0), fwd)
    left = normalize(left)
    cam_up = np.cross(fwd, left)
    return np.stack([fwd, left, cam_up], axis=1)
