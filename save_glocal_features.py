import os
import pickle
import torch

# Load the full embeddings file
with open("features/things/embeddings/model_features_per_source.pkl", "rb") as f:
    all_data = pickle.load(f)

source = "custom"  # change if needed
models = all_data[source].keys()

for model_name in models:
    model_data = all_data[source][model_name]
    features = model_data.get("penultimate")

    if features is None:
        print(f"No penultimate features found for model: {model_name}")
        continue

    features_tensor = torch.tensor(features)

    save_path = os.path.join(
        "features", "things", "embeddings", source, model_name, "penultimate", "train"
    )
    os.makedirs(save_path, exist_ok=True)

    torch.save(features_tensor, os.path.join(save_path, "features.pt"))
    print(f"Saved features.pt for {model_name}")
