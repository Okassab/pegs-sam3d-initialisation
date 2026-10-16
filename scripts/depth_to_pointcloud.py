"""
Back-project a masked depth image into a metric point cloud.

Reads one camera from the PEGS dataset, loads its depth image and the
object mask, and selects the pixels that are both on the object and have
a valid depth reading.
"""

import json
import cv2
import numpy as np

# -------------------------- configurations --------------------------

DATASET = "/workspace/pegs-dataset/real/single1_aruco/modelling/static"
MASK_PATH = "/workspace/pegs-sam3d-input/side-mask.png"
OUT_PATH = "/workspace/pegs-sam3d-results/side_points_world.npy"
VIEW = "side"
FLIP = np.diag([1.0, -1.0, -1.0, 1.0]) # Blender/OpenGL -> OpenCV axis flip, as used by PEGS frames.py

# -------------------------- load --------------------------

with open(f"{DATASET}/transforms.json") as f:
  data = json.load(f)

depth = cv2.imread(f"{DATASET}/depth/{VIEW}.png", cv2.IMREAD_UNCHANGED)
mask = cv2.imread(MASK_PATH, cv2.IMREAD_UNCHANGED)

# print([f["name"] for f in data["frames"]])

# -------------------------- camera parameters --------------------------

camera = None
for frames in data['frames']:
  if frames["name"] == VIEW:
    camera = frames
    break

if camera is None:
  raise ValueError(f"No camera named '{VIEW}' in transform.json")

fl_x = camera["fl_x"]
fl_y = camera["fl_y"]
cx = camera["cx"]
cy = camera["cy"]

# -------------------------- select usable pixels --------------------------

is_object = mask == 255
valid = is_object & (depth > 0)

print(f"depth : {depth.shape} {depth.dtype}")
print(f"mask  : {mask.shape} {mask.dtype}, values {np.unique(mask)}")
print(f"object pixels : {is_object.sum()}")
print(f"with valid depth : {valid.sum()}")


# print(selected.keys())
# print(json.dumps(selected, indent=1))


# -------------------------- back-project to camera coordinates --------------------------

v, u = np.nonzero(valid)
Z = depth[valid] * camera["integer_depth_scale"]
X = (u - cx) * Z / fl_x
Y = (v - cy) * Z / fl_y


points_camera = np.stack([X, Y, Z], axis=1)

X_WC = np.array(camera["transform_matrix"], dtype=np.float64)
X_WC_opencv = X_WC @ FLIP

print(f"Z range: {Z.min():.3f} to {Z.max():.3f} m")
print(f"X range: {X.min():.3f} to {X.max():.3f} m")
print(f"Y range: {Y.min():.3f} to {Y.max():.3f} m")
print(f"points_camera: {points_camera.shape}")

# -------------------------- transform to world coordinates --------------------------

ones = np.ones((points_camera.shape[0], 1))
points_h = np.hstack([points_camera, ones]) # (N, 4) homogeneous

points_world = (X_WC_opencv @ points_h.T).T[:, :3] # (N, 3)

print(f"world X: {points_world[:, 0].min():.3f} to {points_world[:, 0].max():.3f} m")
print(f"world Y: {points_world[:, 1].min():.3f} to {points_world[:, 1].max():.3f} m")
print(f"world Z: {points_world[:, 2].min():.3f} to {points_world[:, 2].max():.3f} m")

# -------------------------- save --------------------------

np.save(OUT_PATH, points_world)
print(f"saved {points_world.shape[0]} points to {OUT_PATH}")