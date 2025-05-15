import pickle
import os

# Adjust path for the model you want to process
path = "features/cifar100-coarse/custom/OpenCLIP_ViT-L-14_laion400m_e32/penultimate/features.pkl"

with open(path, "rb") as f:
    data = pickle.load(f)

features = data["penultimate"]
filenames = data["filenames"]

# Assuming: CIFAR-100-coarse has 50,000 train + 10,000 test images in that order
train_feats = features[:50000]
test_feats = features[50000:]
train_filenames = filenames[:50000]
test_filenames = filenames[50000:]

# Save train
train_out = os.path.join(os.path.dirname(path), "train", "embeddings.pkl")
os.makedirs(os.path.dirname(train_out), exist_ok=True)
with open(train_out, "wb") as f:
    pickle.dump({"features": train_feats, "filenames": train_filenames}, f)

# Save test
test_out = os.path.join(os.path.dirname(path), "test", "embeddings.pkl")
os.makedirs(os.path.dirname(test_out), exist_ok=True)
with open(test_out, "wb") as f:
    pickle.dump({"features": test_feats, "filenames": test_filenames}, f)

print(f"Saved {len(train_feats)} training and {len(test_feats)} test features.")
