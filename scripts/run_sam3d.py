"""
Run SAM 3D Objects on one PEGS dataset view.

Saves the Gaussian-splat reconstruction and the predicted local-to-camera
pose (rotation, translation, scale) produced by the pipeline but discarded
in Session 1.
"""

from pathlib import Path
import numpy as np
import json
import sys

sys.path.insert(0, "/workspace/sam-3d-objects/notebook")
from inference import Inference, load_image, load_mask

# VIEW = "side"
VIEW = sys.argv[1] if len(sys.argv) > 1 else "side"
SEED = 42

DATASET = "/workspace/pegs-dataset/real/single1_aruco/modelling/static"
MASK_PATH = f"/workspace/pegs-sam3d-input/{VIEW}-mask.png"
CHECKPOINT = "/workspace/sam-3d-objects/checkpoints/hf/pipeline.yaml"
RESULTS = Path("/workspace/pegs-sam3d-results")


image = load_image(f"{DATASET}/color/{VIEW}.png")
mask = load_mask(MASK_PATH)
assert image.shape[:2] == mask.shape
print(f"Input ready: {image.shape}, {mask.sum()} mask pixels", flush=True)

model = Inference(CHECKPOINT, compile=False)
output = model(image, mask, seed=SEED)

print(f"output keys: {list(output.keys())}", flush=True)
for key in ("rotation", "translation", "scale"):
    if key in output:
        print(f"  {key}: {tuple(output[key].shape)}")

RESULTS.mkdir(parents=True, exist_ok=True)

output["gs"].save_ply(str(RESULTS / f"single1_aruco_{VIEW}_v2.ply"))

# --- raw local geometry ---

gaussian_xyz = output["gaussian"][0].get_xyz.detach().cpu().numpy()
np.save(RESULTS / f"single1_aruco_{VIEW}_gaussian_xyz.npy", gaussian_xyz)
print(f"gaussian xyz: {gaussian_xyz.shape}")


pose = {
    "view": VIEW,
    "seed": SEED,
    "rotation_quat_l2c": output["rotation"].detach().cpu().numpy().tolist(),
    "translation_l2c": output["translation"].detach().cpu().numpy().tolist(),
    "scale_l2c": output["scale"].detach().cpu().numpy().tolist(),
}

with open(RESULTS / f"single1_aruco_{VIEW}_pose.json", "w") as f:
    json.dump(pose, f, indent=2)

print("saved reconstruction and pose", flush=True)
