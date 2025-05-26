import torch
import numpy as np
import os

ckpt_path = "transforms_reduction/trainset_50pct/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"
output_dir = os.path.dirname(ckpt_path)
out_path = os.path.join(output_dir, "new_test_transform.npz")

ckpt = torch.load(ckpt_path, map_location="cpu")
state_dict = ckpt["state_dict"]

W = state_dict["transform_w"].detach().numpy()

if "transform_b" in state_dict:
    b = state_dict["transform_b"].detach().numpy()
else:
    b = np.zeros(W.shape[0])

mean = ckpt.get("mean", 0.0)
std = ckpt.get("std", 1.0)

print(f"Saving transform: W={W.shape}, b={b.shape}, mean={mean}, std={std}")
np.savez(out_path, weights=W, bias=b, mean=mean, std=std)
