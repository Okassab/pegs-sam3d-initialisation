"""
Compare SAM 3D's predicted pose against the depth measurement.

Applies the predicted local-to-camera transform (rotation, scale,
translation) to the SAM 3D reconstruction, then measures how closely the
result matches the point cloud back-projected from the depth image.
"""


import cv2
import sys
import json
import itertools
import numpy as np
from helper import read_ply, quat_to_matrix, silhouette, iou, align_centroid

# --- configurations ---

RESULTS = "/workspace/pegs-sam3d-results"
# VIEW = "side"
VIEW = sys.argv[1] if len(sys.argv) > 1 else "side"


PLY_PATH = f"{RESULTS}/single1_aruco_{VIEW}_v2.ply"
POSE_PATH = f"{RESULTS}/single1_aruco_{VIEW}_pose.json"
DEPTH_POINTS_PATH = f"{RESULTS}/{VIEW}_points_camera.npy"
DATASET = "/workspace/pegs-dataset/real/single1_aruco/modelling/static"
MASK_PATH = f"/workspace/pegs-sam3d-input/{VIEW}-mask.png"

# --- load ---

data, _, _ = read_ply(PLY_PATH)
points_local = np.stack([data["x"], data["y"], data["z"]], axis=1)

with open(POSE_PATH) as f:
    pose = json.load(f)

quat = np.array(pose["rotation_quat_l2c"][0])
trans = np.array(pose["translation_l2c"][0])
scale = float(pose["scale_l2c"][0][0])

points_depth = np.load(DEPTH_POINTS_PATH)

print(f"local points : {points_local.shape}")
print(f"depth points : {points_depth.shape}")
print(f"quat  : {quat}")
print(f"trans : {trans}")
print(f"scale : {scale}")

# --- camera and mask ---

with open(f"{DATASET}/transforms.json") as f:
    camera = next(c for c in json.load(f)["frames"] if c["name"] == VIEW)

fl_x, fl_y = camera["fl_x"], camera["fl_y"]
cx, cy = camera["cx"], camera["cy"]
height, width = camera["h"], camera["w"]

mask = cv2.imread(MASK_PATH, cv2.IMREAD_UNCHANGED) == 255
print(f"mask: {mask.shape}, {mask.sum()} object pixels")


# --- verify rotation conversion ---
for order in ("wxyz", "xyzw"):
    R = quat_to_matrix(quat, order)
    print(f"{order}: det = {np.linalg.det(R):.6f}")

# A valid rotation matrix has determinant 1. Both orderings should pass;
# this checks the conversion, not which ordering is correct.

# --- search all axis-aligned conventions ---

def signed_permutations():
    """All 48 signed axis permutations: every 90 degree rotation and reflection."""
    mats = {}
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            M = np.zeros((3, 3))
            for row, (col, s) in enumerate(zip(perm, signs)):
                M[row, col] = s
            name = "".join(("-" if s < 0 else "+") + "xyz"[c] for c, s in zip(perm, signs))
            mats[name] = M
    return mats


R_base = quat_to_matrix(quat, "wxyz")

results = []
for label, R in (("R", R_base), ("R^T", R_base.T)):
    for name, F in signed_permutations().items():
        pts = (scale * (points_local @ R)) @ F.T + trans @ F.T
        sil = silhouette(pts, fl_x, fl_y, cx, cy, height, width)
        if sil.sum() == 0:
            continue
        raw = iou(sil, mask)
        aligned = iou(align_centroid(sil, mask), mask)
        results.append((aligned, raw, label, name, int(sil.sum())))

results.sort(reverse=True)
print(f"\n{'aligned':>8} {'raw':>7} {'R':6} {'axes':8} {'px':>8}")
for aligned, raw, label, name, px in results[:10]:
    print(f"{aligned:8.3f} {raw:7.3f} {label:6} {name:8} {px:8d}")

# --- save overlay figure ---

best_aligned, best_raw, best_label, best_name, _ = results[0]
R = R_base.T if best_label == "R^T" else R_base
F = signed_permutations()[best_name]
pts = (scale * (points_local @ R)) @ F.T + trans @ F.T
sil = silhouette(pts, fl_x, fl_y, cx, cy, height, width)

overlay = np.zeros((height, width, 3), dtype=np.uint8)
overlay[mask] = (0, 0, 255)          # mask only: red
overlay[sil] = (0, 255, 0)           # silhouette only: green
overlay[mask & sil] = (0, 255, 255)  # both: yellow

cv2.imwrite(f"{RESULTS}/{VIEW}_overlay.png", overlay)
print(f"saved overlay ({best_label}, {best_name}, raw {best_raw:.3f}, aligned {best_aligned:.3f})")

# --- aligned overlay figure ---

sil_aligned = align_centroid(sil, mask)

overlay_aligned = np.zeros((height, width, 3), dtype=np.uint8)
overlay_aligned[mask] = (0, 0, 255)
overlay_aligned[sil_aligned] = (0, 255, 0)
overlay_aligned[mask & sil_aligned] = (0, 255, 255)

cv2.imwrite(f"{RESULTS}/{VIEW}_overlay_aligned.png", overlay_aligned)
print(f"saved aligned overlay")

