import pickle
import os
from collections import defaultdict

# Define where each model's features live and how they should be repacked
model_sources = [
    {
        "model": "CLIP_RN50",
        "source": "custom",
        "module": "penultimate",
        "path": "features/cifar100/custom/official_clip/penultimate/features.pkl"
    },
    {
        "model": "OpenCLIP_ViT-L-14_laion2b_s32b_b82k",
        "source": "custom",
        "module": "penultimate",
        "path": "features/cifar100/custom/openclip_laion2b/penultimate/features.pkl"
    },
    {
        "model": "OpenCLIP_ViT-L-14_laion400m_e32",
        "source": "custom",
        "module": "penultimate",
        "path": "features/cifar100/custom/openclip_laion400m/penultimate/features.pkl"
    },
]

# Final structure
features = defaultdict(lambda: defaultdict(dict))

for entry in model_sources:
    print(f"Loading: {entry['path']}")
    with open(entry["path"], "rb") as f:
        data = pickle.load(f)
        features[entry["source"]][entry["model"]][entry["module"]] = data

# Make sure the embeddings directory exists
out_path = "results/things/penultimate/penultimate/embeddings"
os.makedirs(out_path, exist_ok=True)

# Save all together
with open(os.path.join(out_path, "features.pkl"), "wb") as f:
    pickle.dump(dict(features), f)

print(f"\n✅ Merged all model features and saved to {os.path.join(out_path, 'features.pkl')}")
