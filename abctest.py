import numpy as np
import hashlib
from sklearn.metrics.pairwise import cosine_similarity
import re

# Define your transforms

paths = [
    ("alpha: 0.05", "transforms2/trainset_10pct/400m/alpha_0.05/transform.npz"),
    ("alpha: 0.1", "transforms2/trainset_10pct/400m/alpha_0.1/transform.npz"),
    ("alpha: 0.25", "transforms2/trainset_10pct/400m/alpha_0.25/transform.npz"),
    ("alpha: 0.5", "transforms2/trainset_10pct/400m/alpha_0.5/transform.npz"),
    ("0.01pct", "transforms3/0.01pct/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"),
    ("0.01pct 0.5aplha", "transforms3/0.01pct/alpha0.5/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"),
    ("1pct", "transforms3/1pct/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"),
    ("test trans 90", "transforms3/90pct/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"),
    ("random trans","transforms3/random/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"),
    ("original transform before experiments", "transforms/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"),
    
]

ws = []
labels = []

for label, path in paths:
    data = np.load(path)
    W = data["weights"].astype(np.float32).reshape(1, -1)  
    ws.append(W)
    labels.append(label)

# Stack and compute cosine similarities
W_matrix = np.vstack(ws)
cos_sim_matrix = cosine_similarity(W_matrix)


# Extract the relevant part from the first path
match = re.search(r"/(trainset_[^/]+)/(\d+m)/", paths[0][1])
if match:
    trainset = match.group(1)
    size = match.group(2)
else:
    print("Could not extract trainset and size from path.")
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



