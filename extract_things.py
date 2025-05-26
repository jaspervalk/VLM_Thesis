import os
import pickle
import torch
import numpy as np
from thingsvision import get_extractor
from thingsvision.utils.data import DataLoader
from data import load_dataset  # assumes you have this from your repo
from torchvision.transforms import Compose, Lambda

# ==== Config ====
DATA_ROOT = "./data/things"  # path to THINGS dataset
OUTPUT_DIR = "./features/things/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate"
MODEL_NAME = "OpenCLIP_ViT-L-14_laion2b_s32b_b82k"
SOURCE = "custom"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
MODULE_NAME = "visual.proj"  #  final projection layer = 1024-dim

# ==== Load extractor ====
extractor = get_extractor(
    model_name="OpenCLIP",
    source=SOURCE,
    device=DEVICE,
    pretrained=True,
    model_parameters=dict(variant="ViT-L-14", dataset="laion2b_s32b_b82k")
)

# ==== Get transform ====
transform = extractor.get_transformations()


# ==== Load dataset ====
dataset = load_dataset(
    name="things",
    data_dir=DATA_ROOT,
    transform=Compose([Lambda(lambda img: img.convert("RGB")), transform]),
    return_label=False,
)

# ==== Dataloader ====
loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, backend=extractor.get_backend())

# ==== Extract features ====
features = extractor.extract_features(
    batches=loader,
    module_name=MODULE_NAME,
    flatten_acts=True,
    output_type="np"
)

# ==== Get filenames ====
try:
    filenames = np.array([os.path.splitext(os.path.basename(p))[0] for p in dataset.image_paths])
except AttributeError:
    filenames = np.array([f"img_{i:05d}" for i in range(len(dataset))])

# ==== Save ====
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, "features.pkl")

with open(out_path, "wb") as f:
    pickle.dump({"penultimate": features, "filenames": filenames}, f)

print(f"Saved features with shape: {features.shape} to {out_path}")
