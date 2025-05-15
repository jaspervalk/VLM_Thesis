import pickle, os

source = "custom"
model = "OpenCLIP_ViT-L-14_laion2b_s32b_b82k"
module = "penultimate"
feature_path = "features/cifar100/custom/openclip_laion2b/penultimate/features.pkl"
output_path = "results/things/penultimate/penultimate/embeddings/features.pkl"

with open(feature_path, "rb") as f:
    feats = pickle.load(f)

nested = {source: {model: {module: feats}}}
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "wb") as f:
    pickle.dump(nested, f)

print(f"✅ Saved {model} features to {output_path}")
