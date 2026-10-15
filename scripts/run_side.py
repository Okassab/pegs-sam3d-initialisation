from pathlib import Path
import sys

sys.path.insert(0, "/workspace/sam-3d-objects/notebook")
from inference import Inference, load_image, load_mask

image = load_image(
    "/workspace/pegs-dataset/real/single1_aruco/modelling/static/color/side.png"
)
mask = load_mask("/workspace/pegs-sam3d-input/side-mask.png")
assert image.shape[:2] == mask.shape
print(f"Input ready: {image.shape}, {mask.sum()} mask pixels", flush=True)

model = Inference(
    "/workspace/sam-3d-objects/checkpoints/hf/pipeline.yaml",
    compile=False,
)
output = model(image, mask, seed=42)

result = Path("/workspace/pegs-sam3d-results/single1_aruco_side.ply")
result.parent.mkdir(parents=True, exist_ok=True)
output["gs"].save_ply(str(result))
print(f"Saved reconstruction: {result}", flush=True)
