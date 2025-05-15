import pickle
import torch
import os
from sklearn.model_selection import train_test_split

# Load
with open("./features/things/custom/CLIP_ViT-L-14_WIT/penultimate/features.pkl", "rb") as f:
    data = pickle.load(f)

features = torch.tensor(data["penultimate"])
filenames = data["filenames"]

# Split
train_feats, val_feats = train_test_split(features, test_size=0.1, random_state=42)

# Save
base_path = "./features/things/custom/CLIP_ViT-L-14_WIT/penultimate"
os.makedirs(f"{base_path}/val", exist_ok=True)
torch.save(val_feats, f"{base_path}/val/features.pt")

print("✅ Saved val/features.pt for gLocal training.")
