import numpy as np
import hashlib
import os

path = "transforms_check/logs/trainset_0_01pct_testtransform/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/transform.npz"

data = np.load(path)
W = data["W"]
print("W shape:", W.shape)
print("W norm:", np.linalg.norm(W))
print("W sum:", W.sum())
print("Is W close to identity?", np.allclose(W, np.eye(W.shape[0]), atol=1e-2))
print("Is W close to zero?", np.allclose(W, 0, atol=1e-2))
W_bytes = W.tobytes()

print("W hash:", hashlib.md5(W_bytes).hexdigest())

W = np.load('transforms_check/logs/trainset_0_01pct_testtransform/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/transform.npz')['W']
f = np.load('transforms_check/logs/trainset_0_01pct_testtransform/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/transform.npz')
print("Keys in .npz file:", f.files)
x = np.random.randn(768)
x_norm = np.linalg.norm(x)

x_proj = W @ x
cos_sim = np.dot(x, x_proj) / (np.linalg.norm(x_proj) * x_norm)

print("Cosine similarity between x and W*x:", cos_sim)
