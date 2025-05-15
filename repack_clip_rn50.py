import pickle, os

source = "custom"
model = "CLIP_RN50"
module = "penultimate"
feature_path = "features/cifar100/custom/official_clip/penultimate/features.pkl"
output_path = "results/things/penultimate/penultimate/embeddings/features.pkl"

with open(feature_path, "rb") as f:
    feats = pickle.load(f)

nested = {source: {model: {module: feats}}}
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "wb") as f:
    pickle.dump(nested, f)

print(f"✅ Saved {model} features to {output_path}")
