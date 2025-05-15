import numpy as np

# Load the transform
path = "./transforms/results/custom/CLIP_ViT-L-14_WIT/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"
transform = np.load(path, allow_pickle=True)

weights = transform["weights"]
bias = transform["bias"] if "bias" in transform else None

# Check and fix scalar NaN bias
if bias is not None:
    if np.isscalar(bias) and np.isnan(bias):
        print("Bias is scalar NaN — replacing with 0.0")
        bias = np.float32(0.0)
    elif isinstance(bias, np.ndarray):
        bias = bias.astype(np.float32)
        if np.isnan(bias).any():
            print("Bias array contains NaNs — replacing them with 0.0")
            bias = np.nan_to_num(bias)

# Save clean transform
np.savez_compressed(
    path,
    weights=weights.astype(np.float32),
    bias=bias
)

print("Saved cleaned transform.")
