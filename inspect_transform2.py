import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Define new transforms with descriptive labels
transform_paths = [
    ("0.01% (400m)", "transforms2/trainset_0_01pct/400m/custom/OpenCLIP_ViT-L-14_laion400m_e32/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"),
    ("0.01% (2b)",   "transforms2/trainset_0_01pct/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"),
    ("1% (400m)",    "transforms2/trainset_01pct/400m/custom/OpenCLIP_ViT-L-14_laion400m_e32/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"),
    ("1% (2b)",      "transforms2/trainset_01pct/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"),
    ("random W", "transforms2/trainset_0_01pct/400m/random_transform.npz"),
]

ws = []
labels = []

for label, path in transform_paths:
    data = np.load(path)
    W = data["weights"].astype(np.float32).reshape(1, -1)
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
