"""
Shared helpers for the PEGS / SAM 3D scripts.

Contains PLY reading and quaternion conversion, both used by more than
one script.
"""

import numpy as np


def read_ply(path):
    """Parse a binary little-endian PLY into a numpy structured array."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break

        n_vertices = None
        properties = []
        for line in header:
            if line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
            elif line.startswith("property float"):
                properties.append(line.split()[-1])

        if n_vertices is None:
            raise ValueError("No 'element vertex' line found in header")

        fmt = [l for l in header if l.startswith("format")][0]
        if "binary_little_endian" not in fmt:
            raise ValueError(f"Only binary_little_endian is supported, got: {fmt}")

        dtype = np.dtype([(p, "<f4") for p in properties])
        data = np.frombuffer(f.read(n_vertices * dtype.itemsize), dtype=dtype)

    return data, properties, n_vertices


def quat_to_matrix(q, order="wxyz"):
    """Convert a unit quaternion to a 3x3 rotation matrix."""
    if order == "wxyz":
        w, x, y, z = q
    elif order == "xyzw":
        x, y, z, w = q
    else:
        raise ValueError(f"unknown quaternion order: {order}")

    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])

# --- silhouette projection ---

def silhouette(points, fl_x, fl_y, cx, cy, height, width):
    """Project camera-frame points into the image and rasterise to a boolean mask."""
    # points behind the camera do not project; discard them
    p = points[points[:, 2] > 0]

    u = np.round(fl_x * p[:, 0] / p[:, 2] + cx).astype(int)
    v = np.round(fl_y * p[:, 1] / p[:, 2] + cy).astype(int)

    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)

    image = np.zeros((height, width), dtype=bool)
    image[v[inside], u[inside]] = True
    return image


def iou(a, b):
    """Intersection over union of two boolean masks."""
    return (a & b).sum() / (a | b).sum()
