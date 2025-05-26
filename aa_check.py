import numpy as np
import hashlib

def load_w_and_hash(path):
    data = np.load(path)
    if 'W' not in data:
        raise KeyError(f"'W' not found in {path}. Keys are: {list(data.keys())}")
    W = data['W']
    w_hash = hashlib.md5(W.astype(np.float32).tobytes()).hexdigest()
    return W, w_hash

# Paths to the transform files
path_001 = "transforms_check/logs/trainset_0_01pct_testtransform/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"
path_50 = "transforms_reduction/trainset_50pct/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"

# Load and hash both
W_001, hash_001 = load_w_and_hash(path_001)
W_50, hash_50 = load_w_and_hash(path_50)

# Output comparison
print("=== Transform Comparison ===")
print(f"0.01% W shape: {W_001.shape}, hash: {hash_001}")
print(f"50%   W shape: {W_50.shape}, hash: {hash_50}")
print(f"Hashes are {'IDENTICAL' if hash_001 == hash_50 else 'DIFFERENT'}")
