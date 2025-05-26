import numpy as np
import hashlib
from sklearn.metrics.pairwise import cosine_similarity

# Define your transforms
transform_paths = [
    ("random", "transforms_check/logs/random_transform/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"),
    ("test 0.01%", "transforms_check/logs/trainset_0_01pct_testtransform/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"),
    ("0.1%", "transforms_reduction/trainset_0_1pct/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"),
    ("1%", "transforms_reduction/trainset_01pct/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"),
    ("10%", "transforms_reduction/trainset_10pct/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"),
    ("25%", "transforms_reduction/trainset_25pct/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"),
    ("50%", "transforms_reduction/trainset_50pct/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"),
    ("100% test", "transforms_reduction/trainset_100pct/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"),
    ("original transform before experiments", "transforms/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"),


]

ws = []
labels = []

for label, path in transform_paths:
    data = np.load(path)
    W = data["W"].astype(np.float32).reshape(1, -1)  
    ws.append(W)
    labels.append(label)

# Stack and compute cosine similarities
W_matrix = np.vstack(ws)
cos_sim_matrix = cosine_similarity(W_matrix)



print("Cosine similarity between W matrices:\n")
print("".ljust(15), end="")
for l in labels:
    print(l.ljust(15), end="")
print()

for i, row in enumerate(cos_sim_matrix):
    print(labels[i].ljust(15), end="")
    for sim in row:
        print(f"{sim:.4f}".ljust(15), end="")
    print()

