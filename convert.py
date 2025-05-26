import pickle
import torch
import os
import numpy as np

# Load original combined features
pkl_path = "./features/things/custom/OpenCLIP_ViT-L-14_laion400m_e32/penultimate/features.pkl"
out_root = os.path.dirname(pkl_path)

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

features = data["penultimate"]
filenames = data["filenames"]

print(f"Loaded: {features.shape} features")

# Identify train vs val by filename suffix
filenames = np.array(filenames)
is_train = np.char.endswith(filenames, "b")
is_val   = np.char.endswith(filenames, "s")

train_features = features[is_train]
val_features   = features[is_val]

print(f"Train features: {train_features.shape}")
print(f"Val features:   {val_features.shape}")

# Save to .pt files
os.makedirs(os.path.join(out_root, "train"), exist_ok=True)
os.makedirs(os.path.join(out_root, "val"), exist_ok=True)

torch.save({"features": train_features}, os.path.join(out_root, "train", "features.pt"))
torch.save({"features": val_features}, os.path.join(out_root, "val", "features.pt"))

print("Saved train/val features.pt")
