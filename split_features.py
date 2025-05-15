import os
import pickle
import torch
from sklearn.model_selection import train_test_split

# === Settings ===
datasets = ["cifar100", "cifar100-coarse"]
models = ["openclip_laion2b", "openclip_laion400m", "official_clip"]
input_root = "features"
output_root = "aux_features_root"
source = "custom"
module = "penultimate"
test_size = 0.2
seed = 42

for dataset in datasets:
    for model in models:
        input_path = os.path.join(input_root, dataset, source, model, module, "features.pkl")
        output_dir = os.path.join(output_root, dataset, model)

        print(f"📂 Loading: {input_path}")
        if not os.path.isfile(input_path):
            print(f"❌ Missing: {input_path} — skipping.")
            continue

        # Load the pkl file
        with open(input_path, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, dict) or "penultimate" not in data:
            print(f"⚠️ Unexpected format in {input_path} — skipping.")
            continue

        features = data["penultimate"]
        print(f"✅ Loaded features with shape: {features.shape}")

        # Train-val split
        train_feats, val_feats = train_test_split(
            features.astype("float32"), test_size=test_size, random_state=seed
        )

        # Output paths
        train_out = os.path.join(output_dir, "train.pt")
        val_out = os.path.join(output_dir, "val.pt")

        os.makedirs(output_dir, exist_ok=True)
        torch.save(torch.tensor(train_feats), train_out)
        torch.save(torch.tensor(val_feats), val_out)

        print(f"💾 Saved to: {train_out} and {val_out}\n")

print("✅ Done converting all feature sets!")
