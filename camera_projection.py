"""Pinhole camera model: project a 3D triangle mesh (STL) onto a virtual camera's image plane.

Steps
1. Load the STL mesh (a human figure, Z-up, about 1.8 units tall).
2. Build a scene with four copies of the figure placed 1 unit apart along the world X axis.
3. Build the extrinsic matrix P_extr = [R | t; 0 1] with R = Ry(-90 deg) @ Rx(90 deg)
   and t = -R @ C for a camera centre C = (-2, 0.8, 0). Row 3 of R is the world +X axis, so the
   camera looks along +X towards the figures.
4. Transform all points to camera coordinates, Q' = P_extr @ Q (homogeneous, 4 x N), keep the
   points in front of the near plane (z' > 0.1) and divide by depth: u = x'/z', v = y'/z'.
   The intrinsic matrix is the identity (focal length 1), so (u, v) are normalised image coordinates.
5. Save three figures: the original mesh, the scene, and the projected image. The projected image
   is drawn twice: as points (as in the course task) and as filled triangles sorted by depth
   (painter's algorithm), which shows which figure hides which.

Usage:
    python camera_projection.py                         # default task settings, writes figures/
    python camera_projection.py --camera -3 0.8 0.2     # move the camera
    python camera_projection.py --copies 6 --spacing 0.8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from stl import mesh as stl_mesh

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------- geometry
def rotation_x(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_y(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def extrinsic_matrix(R: np.ndarray, C: np.ndarray) -> np.ndarray:
    """4 x 4 world-to-camera matrix [R | t; 0 0 0 1] with t = -R C (C = camera centre in world)."""
    P = np.eye(4)
    P[:3, :3] = R
    P[:3, 3] = -R @ np.asarray(C, dtype=float).reshape(3) + 0.0  # + 0.0 turns -0.0 into 0.0
    return P


def to_camera(points: np.ndarray, P_extr: np.ndarray) -> np.ndarray:
    """N x 3 world points -> N x 3 camera coordinates (x', y', z') via homogeneous Q' = P Q."""
    Q = np.vstack([points.T, np.ones(len(points))])  # 4 x N
    return (P_extr @ Q)[:3].T


def project(points_cam: np.ndarray, near: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Perspective division of camera coordinates. Returns (uv, mask of points with z' > near)."""
    mask = points_cam[:, 2] > near
    uv = np.full((len(points_cam), 2), np.nan)
    uv[mask] = points_cam[mask, :2] / points_cam[mask, 2:3]
    return uv, mask


# ---------------------------------------------------------------- scene
def load_triangles(path: Path) -> np.ndarray:
    """STL triangles as an (n_triangles, 3, 3) array: 3 corners x (X, Y, Z)."""
    return np.asarray(stl_mesh.Mesh.from_file(str(path)).vectors, dtype=float)


def make_scene(triangles: np.ndarray, copies: int = 4, spacing: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Copies of the mesh shifted by i * spacing along X. Returns (triangles, copy index per triangle)."""
    shifted = [triangles + np.array([i * spacing, 0.0, 0.0]) for i in range(copies)]
    ids = np.repeat(np.arange(copies), len(triangles))
    return np.concatenate(shifted), ids


# ---------------------------------------------------------------- figures
def plot_mesh_3d(points, path, title, sample=10, limits=None, aspect=(1, 1, 1), view=None):
    fig = plt.figure(figsize=(6, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    p = points[::sample]
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=0.5, c=p[:, 0], cmap="viridis")
    if limits:
        ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1]); ax.set_zlim(*limits[2])
    ax.set_box_aspect(aspect)
    if view:
        ax.view_init(*view)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_projection(tri_cam, copy_ids, path, near=0.1):
    """Left: projected vertices. Right: triangles filled in far-to-near order, shaded by facing angle."""
    keep = (tri_cam[:, :, 2] > near).all(axis=1)  # drop triangles that cross the near plane
    tri_cam, copy_ids = tri_cam[keep], copy_ids[keep]
    uv = tri_cam[:, :, :2] / tri_cam[:, :, 2:3]

    # Lambert-like shading from the angle between the face normal and the viewing ray.
    normals = np.cross(tri_cam[:, 1] - tri_cam[:, 0], tri_cam[:, 2] - tri_cam[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    rays = tri_cam.mean(axis=1)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    shade = 0.25 + 0.75 * np.abs((normals * rays).sum(axis=1))

    order = np.argsort(-tri_cam[:, :, 2].mean(axis=1))  # far first, near last
    palette = plt.get_cmap("viridis")(np.linspace(0.1, 0.85, copy_ids.max() + 1))[:, :3]
    colours = palette[copy_ids] * shade[:, None]

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 6.5), sharex=True, sharey=True)
    pts = uv.reshape(-1, 2)
    axes[0].plot(pts[:, 0], pts[:, 1], "k.", markersize=0.3)
    axes[0].set_title("Projected vertices")
    axes[1].add_collection(PolyCollection(uv[order], facecolors=colours[order], edgecolors="none"))
    axes[1].set_title("Filled triangles, drawn far to near")
    for ax in axes:
        ax.set_xlabel("u = x'/z'"); ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.autoscale_view()
    axes[0].set_ylabel("v = y'/z'")
    axes[0].invert_yaxis()  # image convention: v grows downwards (camera y' = world -Z)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stl", default=str(ROOT / "man_small.stl"))
    p.add_argument("--copies", type=int, default=4)
    p.add_argument("--spacing", type=float, default=1.0)
    p.add_argument("--camera", type=float, nargs=3, default=[-2.0, 0.8, 0.0], metavar=("X", "Y", "Z"))
    p.add_argument("--near", type=float, default=0.1)
    p.add_argument("--out", default=str(ROOT / "figures"))
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    triangles = load_triangles(Path(args.stl))
    corners = triangles.reshape(-1, 3)
    unique = np.unique(corners, axis=0)
    scene, ids = make_scene(triangles, args.copies, args.spacing)

    R = np.round(rotation_y(-90) @ rotation_x(90), 12) + 0.0  # exact 0/1 entries, no -0.0
    C = np.array(args.camera)
    P = extrinsic_matrix(R, C)
    scene_cam = to_camera(scene.reshape(-1, 3), P).reshape(scene.shape)
    _, mask = project(scene_cam.reshape(-1, 3), args.near)

    # Apparent height of each figure in the image (v range), to show the perspective effect.
    heights = []
    for i in range(args.copies):
        cam = scene_cam[ids == i].reshape(-1, 3)
        uv, m = project(cam, args.near)
        heights.append({"copy": i, "mean_depth": round(float(cam[m, 2].mean()), 3),
                        "image_height": round(float(np.ptp(uv[m, 1])), 3) if m.any() else None})

    plot_mesh_3d(corners, out / "fig1_mesh.png", "Original mesh", sample=5,
                 limits=[(-0.8, 0.8)] * 3)
    plot_mesh_3d(scene.reshape(-1, 3), out / "fig2_scene.png", f"Scene: {args.copies} copies",
                 sample=20, limits=[(-1, args.copies - 0.5), (-1, 1), (-1, 1)],
                 aspect=(1, 0.5, 0.5), view=(20, -60))
    plot_projection(scene_cam, ids, out / "fig3_projection.png", args.near)

    np.set_printoptions(precision=3, suppress=True)
    print(f"Mesh: {len(triangles)} triangles, {len(corners)} corner entries, {len(unique)} unique vertices")
    print(f"Bounding box: min {corners.min(0)}, max {corners.max(0)}")
    print("R = Ry(-90) @ Rx(90) =\n", R.round(6))
    print("t = -R C =", P[:3, 3].round(6))
    print("P_extr =\n", P.round(6))
    print(f"Points in front of the near plane: {mask.sum()} / {mask.size}")
    for h in heights:
        print(f"  copy {h['copy']}: mean depth {h['mean_depth']}, image height {h['image_height']}")

    summary = {"camera_centre": C.tolist(), "R": R.round(6).tolist(), "t": P[:3, 3].round(6).tolist(),
               "triangles": int(len(triangles)), "unique_vertices": int(len(unique)),
               "points_in_front": int(mask.sum()), "points_total": int(mask.size), "figures": heights}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
