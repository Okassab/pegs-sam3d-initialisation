"""
Render a SAM 3D reconstruction from several viewpoints.

Produces figures showing the reconstructed object from angles other than the
one it was reconstructed from, including surfaces that no camera observed.

Usage:
    python render_views.py [view]
"""

import sys
import numpy as np
import cv2
from helper import read_ply

# --- configuration ---

VIEW = sys.argv[1] if len(sys.argv) > 1 else "side"
RESULTS = "/workspace/pegs-sam3d-results"
PLY_PATH = f"{RESULTS}/single1_aruco_{VIEW}_v2.ply"

SIZE = 800            # output image is SIZE x SIZE
MARGIN = 0.85         # fraction of the frame the object fills
ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
SH_C0 = 0.28209479177387814   # zeroth order spherical harmonic coefficient
BACKGROUND = 255      # white

# --- load ---

data, _, n = read_ply(PLY_PATH)

xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)

# 3DGS stores colour as spherical harmonic DC terms and opacity as a logit
rgb = SH_C0 * np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=1) + 0.5
rgb = np.clip(rgb, 0.0, 1.0)
alpha = 1.0 / (1.0 + np.exp(-data["opacity"]))

# discard near-transparent primitives, which contribute noise rather than surface
keep = alpha > 0.5
xyz, rgb = xyz[keep], rgb[keep]
print(f"{VIEW}: {n:,} gaussians, {keep.sum():,} kept after opacity filter")

# centre on the object and normalise so it fills the frame at any rotation
xyz = xyz - xyz.mean(axis=0)
radius = np.linalg.norm(xyz, axis=1).max()
xyz = xyz / radius

# --- render ---

def rotate_y(points, degrees):
    """Rotate points about the vertical axis."""
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return points @ R.T


for angle in ANGLES:
    p = rotate_y(xyz, angle)

    # orthographic projection; v is negated because image rows run downward
    scale = (SIZE / 2) * MARGIN
    u = np.round(SIZE / 2 + scale * p[:, 0]).astype(int)
    v = np.round(SIZE / 2 - scale * p[:, 1]).astype(int)

    inside = (u >= 0) & (u < SIZE) & (v >= 0) & (v < SIZE)
    u, v, depth, colour = u[inside], v[inside], p[inside, 2], rgb[inside]

    # painter's algorithm: draw far points first so near ones overwrite them
    order = np.argsort(-depth)
    u, v, colour = u[order], v[order], colour[order]

    image = np.full((SIZE, SIZE, 3), BACKGROUND, dtype=np.uint8)
    image[v, u] = (colour[:, ::-1] * 255).astype(np.uint8)  # RGB -> BGR for OpenCV

    out_path = f"{RESULTS}/{VIEW}_render_{angle:03d}.png"
    cv2.imwrite(out_path, image)
    print(f"  {angle:3d} deg -> {out_path}")
