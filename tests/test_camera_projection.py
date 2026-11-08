import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_projection import (extrinsic_matrix, load_triangles, make_scene, project, rotation_x,  # noqa: E402
                               rotation_y, to_camera)

R = rotation_y(-90) @ rotation_x(90)
C = np.array([-2.0, 0.8, 0.0])
P = extrinsic_matrix(R, C)


def test_rotation_is_proper_and_matches_hand_derivation():
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.0)  # a rotation, not a reflection
    assert np.allclose(R, [[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    assert np.allclose(P[:3, 3], [0.8, 0.0, 2.0])  # t = -R C


def test_camera_centre_maps_to_origin_and_looks_along_world_x():
    assert np.allclose(to_camera(C[None], P), 0)
    on_axis = C + np.array([[3.0, 0, 0]])  # 3 units in front of the camera
    cam = to_camera(on_axis, P)
    assert np.allclose(cam, [[0, 0, 3]])
    uv, mask = project(cam)
    assert mask.all() and np.allclose(uv, 0)


def test_perspective_division_and_near_plane():
    cam = np.array([[1.0, 2.0, 4.0], [1.0, 1.0, 0.05], [1.0, 1.0, -2.0]])
    uv, mask = project(cam, near=0.1)
    assert mask.tolist() == [True, False, False]
    assert np.allclose(uv[0], [0.25, 0.5])
    assert np.isnan(uv[1:]).all()


def test_farther_figures_look_smaller():
    tri = load_triangles(Path(__file__).resolve().parents[1] / "man_small.stl")
    scene, ids = make_scene(tri, copies=4, spacing=1.0)
    cam = to_camera(scene.reshape(-1, 3), P).reshape(scene.shape)
    heights = []
    for i in range(4):
        uv, m = project(cam[ids == i].reshape(-1, 3))
        heights.append(np.ptp(uv[m, 1]))
    assert all(a > b for a, b in zip(heights, heights[1:]))
