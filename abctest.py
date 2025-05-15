import pickle

with open("features/cifar100-coarse/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/features.pkl", "rb") as f:
    data = pickle.load(f)

penultimate = data["penultimate"]
print("Keys inside 'penultimate':", len(penultimate))
print("Sample keys:", list(penultimate.keys())[:5])
