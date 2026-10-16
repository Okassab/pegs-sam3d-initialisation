"""
Generate binary object masks from the PEGS dataset instance images.

The instance images encode the object in channel 2 with value 128.
Produces one PNG per view for use as SAM 3D input.
"""

import cv2
import numpy as np

# --- configuration ---

DATASET = "/workspace/pegs-dataset/real/single1_aruco/modelling/static"
OUT_DIR = "/workspace/pegs-sam3d-input"
VIEWS = ["side", "back", "topleft"]
OBJECT_VALUE = 128

# --- generate ---

for view in VIEWS:
    instance = cv2.imread(f"{DATASET}/instance/{view}.png", cv2.IMREAD_UNCHANGED)
    mask = (instance[:, :, 2] == OBJECT_VALUE).astype(np.uint8) * 255
    out_path = f"{OUT_DIR}/{view}-mask.png"
    cv2.imwrite(out_path, mask)
    print(f"{view}: {(mask > 0).sum()} object pixels -> {out_path}")