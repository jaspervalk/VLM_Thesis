import numpy as np

# Load transform with allow_pickle=True
transform_path = "./transforms/results/custom/CLIP_ViT-L-14_WIT/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"
transform = np.load(transform_path, allow_pickle=True)

# Check weights
weights = transform["weights"]
print("Weights contain NaNs:", np.isnan(weights).any())

# Check bias if it exists
if "bias" in transform:
    bias = transform["bias"]
    if isinstance(bias, np.ndarray) and np.issubdtype(bias.dtype, np.number):
        print("Bias contains NaNs:", np.isnan(bias).any())
    else:
        print("Bias is not a numeric array. Type:", type(bias), "Dtype:", getattr(bias, "dtype", "N/A"))
else:
    print("No bias found.")
