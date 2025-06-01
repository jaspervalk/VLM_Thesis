import numpy as np
import hashlib

transform_paths = [
    ("0.01% (400m)", "transforms2/trainset_0_01pct/400m/custom/OpenCLIP_ViT-L-14_laion400m_e32/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"),
    ("0.01% (2b)",   "transforms2/trainset_0_01pct/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"),
    ("1% (400m)",    "transforms2/trainset_01pct/400m/custom/OpenCLIP_ViT-L-14_laion400m_e32/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"),
    ("1% (2b)",      "transforms2/trainset_01pct/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"),
    ("random W", "transforms2/trainset_0_01pct/400m/random_transform.npz")
]

KEY_TO_USE = "weights" 

for label, path in transform_paths:
    print(label)
    data = np.load(path)
    W = data[KEY_TO_USE]
    print("W shape:", W.shape)
    print("W norm:", np.linalg.norm(W))
    print("W sum:", W.sum())
    if W.shape[0] == W.shape[1]:
        print("Is W close to identity?", np.allclose(W, np.eye(W.shape[0]), atol=1e-2))
    else:
        print("Is W close to identity?", "N/A (not square)")
    print("Is W close to zero?", np.allclose(W, 0, atol=1e-2))
    W_bytes = W.tobytes()
    print("W hash:", hashlib.md5(W_bytes).hexdigest())
    print()
