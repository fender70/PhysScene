import math

import numpy as np
import pytest

from physscene import geometry as g


def random_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def test_quat_matrix_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(100):
        q = random_quat(rng)
        q2 = g.matrix_to_quat(g.quat_to_matrix(q))
        assert min(np.abs(q - q2).max(), np.abs(q + q2).max()) < 1e-9


@pytest.mark.parametrize("rot", [(0, 0, 0), (10, 20, 30), (-45, 60, 170), (90, -30, -120)])
def test_ue_rotator_roundtrip(rot):
    m = g.ue_rotator_to_matrix(*rot)
    assert np.allclose(g.ue_rotator_to_matrix(*g.ue_matrix_to_rotator(m)), m, atol=1e-9)


def test_rh_to_ue_mirror():
    # +90 deg yaw in right-handed maps +X onto +Y (RH), which is -Y in Unreal -> yaw -90.
    roll, pitch, yaw = g.quat_to_ue_rotator(g.quat_from_yaw(math.pi / 2))
    assert abs(roll) < 1e-9 and abs(pitch) < 1e-9 and abs(yaw + 90) < 1e-9
    assert g.to_ue_location([1, 2, 3]) == [100, -200, 300]


def test_ue_rotation_consistent_with_points():
    """Rotating a point in RH then converting == converting then rotating in UE."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        q = random_quat(rng)
        p = rng.normal(size=3)
        rh_rot = g.quat_to_matrix(q) @ p
        m_ue = g.ue_rotator_to_matrix(*g.quat_to_ue_rotator(q))
        ue_rot = m_ue @ (np.array(g.to_ue_location(p)) / 100)
        assert np.allclose(np.array(g.to_ue_location(rh_rot)) / 100, ue_rot, atol=1e-9)
        q2 = g.ue_rotator_to_quat(*g.quat_to_ue_rotator(q))
        assert min(np.abs(q2 - q).max(), np.abs(q2 + q).max()) < 1e-7


def test_unwrap():
    out = g.unwrap_degrees([[0, 0, 170], [0, 0, -175], [0, 0, -160]])
    assert np.allclose([r[2] for r in out], [170, 185, 200])


def test_look_at_is_right_handed_rotation():
    m = g.look_at_matrix([2, 1, 1], [0, 0, 0])
    assert np.allclose(m.T @ m, np.eye(3)) and np.isclose(np.linalg.det(m), 1)
    assert np.allclose(m[:, 0], -np.array([2, 1, 1]) / np.linalg.norm([2, 1, 1]))
