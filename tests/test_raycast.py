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


def test_cached_clip_renderer_matches_per_frame_renderer():
    """render_frames caches static geometry; its pixels must equal render_frame's,
    including frames where a moving body vanishes or casts a shadow on a static one."""
    from physscene.preview import render_frame, render_frames

    floor = Primitive("floor", None, np.array([0.0, 0.0, 0.0]), quat_identity(), (0.5, 0.5, 0.5))
    table = Primitive("surface:table", Shape("box", half_extents=(0.6, 0.4, 0.02)), np.array([0.0, 0.0, 0.7]),
                      quat_identity(), (0.6, 0.45, 0.3))
    pillar = Primitive("occluder", Shape("box", half_extents=(0.05, 0.05, 0.15)), np.array([0.1, -0.2, 0.87]),
                       quat_identity(), (0.4, 0.4, 0.43), stencil=2)
    frames = []
    for f in range(6):
        prims = [floor, table, pillar]
        if f != 3:  # the ball vanishes in frame 3
            q = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), 0.3 * f)
            prims.append(Primitive("object", Shape("sphere", radius=0.06), np.array([-0.4 + 0.15 * f, 0.0, 0.78]), q,
                                   (0.8, 0.1, 0.1), stencil=1))
        frames.append(prims)
    from physscene.geometry import quat_mul

    # camera forward is +X: turn it to +Y, then tilt it 20 degrees down
    q = quat_mul(quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2), quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), np.radians(20)))
    cam = CameraSpec(position=[-0.2, -1.6, 1.3], quat=list(q), hfov_deg=55.0, resolution=[96, 64])
    _, st0, _ = render_frame(frames[0], cam, 96, 64)
    assert {0, 1, 2} <= set(np.unique(st0))  # table, ball and pillar are all in view
    for prims, (rgb, st, dp) in zip(frames, render_frames(frames, cam, 96, 64)):
        r0, s0, d0 = render_frame(prims, cam, 96, 64)
        assert np.array_equal(rgb, r0) and np.array_equal(st, s0) and np.array_equal(dp, d0)
