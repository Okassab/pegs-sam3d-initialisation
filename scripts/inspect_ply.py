#!/usr/bin/env python3
"""
Inspect a Gaussian-splat PLY file.

Reports the vertex count, spatial extent, centroid and primitive statistics
without loading the file into a renderer. Used to determine the coordinate
frame and scale of SAM 3D Objects output prior to registration.

Usage:
    python inspect_ply.py /path/to/file.ply
"""

import sys
import numpy as np


def read_ply(path):
    """Parse a binary little-endian PLY into a numpy structured array."""
    with open(path, "rb") as f:
        # --- header is plain ASCII, terminated by "end_header" ---
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


def summarise(data, properties, n_vertices, path):
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)

    print(f"File        : {path}")
    print(f"Vertices    : {n_vertices:,}")
    print(f"Properties  : {', '.join(properties)}")
    print()

    # --- full extent, including any stray outlier primitives ---
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    centre = (lo + hi) / 2.0
    extent = hi - lo

    print("Bounding box (full)")
    for i, axis in enumerate("xyz"):
        print(f"  {axis}: min {lo[i]:+9.4f}   max {hi[i]:+9.4f}   extent {extent[i]:9.4f}")
    print(f"  centre      : ({centre[0]:+.4f}, {centre[1]:+.4f}, {centre[2]:+.4f})")
    print(f"  diagonal    : {np.linalg.norm(extent):.4f}")
    print(f"  centroid    : ({xyz[:, 0].mean():+.4f}, {xyz[:, 1].mean():+.4f}, {xyz[:, 2].mean():+.4f})")
    print()

    # --- robust extent: ignores floaters, which distort the full box ---
    p1, p99 = np.percentile(xyz, 1, axis=0), np.percentile(xyz, 99, axis=0)
    robust = p99 - p1
    print("Bounding box (1st-99th percentile, floater-robust)")
    for i, axis in enumerate("xyz"):
        print(f"  {axis}: min {p1[i]:+9.4f}   max {p99[i]:+9.4f}   extent {robust[i]:9.4f}")
    print(f"  diagonal    : {np.linalg.norm(robust):.4f}")
    print()

    # --- distance from origin tells us whether the frame is object-centred ---
    dist = np.linalg.norm(xyz, axis=1)
    print("Distance from origin")
    print(f"  min {dist.min():.4f}   median {np.median(dist):.4f}   max {dist.max():.4f}")
    print()

    # 3DGS convention stores scale as log and opacity as logit. Both raw and
    # activated values are printed because SAM 3D's convention is unconfirmed.
    if "scale_0" in properties:
        s = np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=1)
        print("Scale (raw / assumed log-space)")
        print(f"  min {s.min():+.4f}   median {np.median(s):+.4f}   max {s.max():+.4f}")
        print(f"  exp(): median {np.exp(np.median(s)):.6f}   max {np.exp(s.max()):.6f}")
        print()

    if "opacity" in properties:
        o = data["opacity"]
        sig = 1.0 / (1.0 + np.exp(-o))
        print("Opacity (raw / assumed logit)")
        print(f"  raw      : min {o.min():+.4f}  median {np.median(o):+.4f}  max {o.max():+.4f}")
        print(f"  sigmoid(): median {np.median(sig):.4f}   fraction > 0.5: {(sig > 0.5).mean():.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    data, properties, n_vertices = read_ply(path)
    summarise(data, properties, n_vertices, path)
