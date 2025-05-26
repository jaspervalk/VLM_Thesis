import numpy as np

npz_path = "transforms2/trainset_0_01pct/transform.npz"
transform = np.load(npz_path, allow_pickle=True)

print(f"\nFile: {npz_path}")
print("Keys:", list(transform.keys()))

if "weights" in transform:
    W = transform["weights"]
    print(f"weights: shape={W.shape}, dtype={W.dtype}")
    print(f"  weights stats: mean={W.mean():.5f}, std={W.std():.5f}, min={W.min():.5f}, max={W.max():.5f}")
elif "W" in transform:
    W = transform["W"]
    print(f"W: shape={W.shape}, dtype={W.dtype}")
    print(f"  W stats: mean={W.mean():.5f}, std={W.std():.5f}, min={W.min():.5f}, max={W.max():.5f}")
if "bias" in transform:
    b = transform["bias"]
    # Convert 0-dim object array to actual object
    if isinstance(b, np.ndarray) and b.dtype == object:
        b = b.item()
    if b is None:
        print("bias: None (no bias used)")
    else:
        print(f"bias: shape={b.shape}, dtype={b.dtype}")
        print(f"  bias stats: mean={b.mean():.5f}, std={b.std():.5f}, min={b.min():.5f}, max={b.max():.5f}")

elif "b" in transform:
    b = transform["b"]
    print(f"b: shape={b.shape}, dtype={b.dtype}")
    print(f"  b stats: mean={b.mean():.5f}, std={b.std():.5f}, min={b.min():.5f}, max={b.max():.5f}")

for key in ["mean", "std"]:
    if key in transform:
        print(f"{key}: shape={transform[key].shape}, value={transform[key]}")
