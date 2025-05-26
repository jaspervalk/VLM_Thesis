import numpy as np
import random

triplet_path = "triplet_dataset/trainset.txt"
out_npy = "triplet_dataset/trainset_01pct.npy"


with open(triplet_path, "r") as f:
    triplets = [list(map(int, line.strip().split())) for line in f]

print(f"Total triplets loaded: {len(triplets)}")

random.seed(42)
subset = random.sample(triplets, int(len(triplets) * 0.01))
subset = np.array(subset)

np.save(out_npy, subset)
print(f"Saved reduced triplets to: {out_npy} — shape: {subset.shape}")
