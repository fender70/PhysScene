import numpy as np

from physscene.catalog import Shape
from physscene.geometry import quat_from_axis_angle, quat_identity
from physscene.raycast import Primitive, cast, intersect, project
from physscene.spec import CameraSpec


def ray(o, d):
    return np.array([o], float), np.array([d], float)


def test_sphere_box_cylinder_hits():
    o, d = ray([-5, 0, 0], [1, 0, 0])
    s = Primitive("s", Shape("sphere", radius=1), np.zeros(3), quat_identity())
    b = Primitive("b", Shape("box", half_extents=(0.5, 1, 1)), np.zeros(3), quat_identity())
    c = Primitive("c", Shape("cylinder", radius=0.3, half_length=1), np.zeros(3), quat_identity())
    assert np.isclose(intersect(o, d, s)[0], 4.0)
    assert np.isclose(intersect(o, d, b)[0], 4.5)
    assert np.isclose(intersect(o, d, c)[0], 4.7)
    # cylinder cap: from above
    o2, d2 = ray([0.1, 0, 5], [0, 0, -1])
    assert np.isclose(intersect(o2, d2, c)[0], 4.0)
    # rotated box (45 deg about z): half diagonal of x-y face is sqrt(0.5^2+1^2)... just check hit
    rb = Primitive("rb", Shape("box", half_extents=(0.5, 0.5, 0.5)), np.zeros(3), quat_from_axis_angle((0, 0, 1), np.pi / 4))
    assert np.isclose(intersect(o, d, rb)[0], 5 - np.sqrt(0.5))


def test_miss_and_nearest():
    o, d = ray([-5, 3, 0], [1, 0, 0])
    s = Primitive("s", Shape("sphere", radius=1), np.zeros(3), quat_identity())
    assert np.isinf(intersect(o, d, s)[0])
    o, d = ray([-5, 0, 0], [1, 0, 0])
    far = Primitive("far", Shape("sphere", radius=1), np.array([3.0, 0, 0]), quat_identity())
    t, i = cast(o, d, [far, s])
    assert i[0] == 1 and np.isclose(t[0], 4)


def test_projection_center():
    from physscene.geometry import look_at_matrix, matrix_to_quat

    cam = CameraSpec([3, 0, 0], matrix_to_quat(look_at_matrix([3, 0, 0], [0, 0, 0])).tolist(), 60.0, [200, 100])
    uv, depth = project(cam, np.array([[0, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]))
    assert np.allclose(uv[0], [100, 50]) and np.isclose(depth[0], 3)
    assert uv[1, 0] > 100  # a camera at +x looking toward -x has -y on its left, so +y projects right
    assert uv[2, 1] < 50  # up projects up
