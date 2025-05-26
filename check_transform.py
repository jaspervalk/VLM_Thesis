import numpy as np

paths = [
    ("original", "transforms/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"),
    ("50pct", "transforms_reduction/trainset_50pct/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"),
    ("random", "transforms_check/logs/random_transform/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"),
    ("test 0.01%", "transforms_check/logs/trainset_0_01pct_testtransform/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"),
    ("0.1%", "transforms_reduction/trainset_0_1pct/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"),
    ("1%", "transforms_reduction/trainset_01pct/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"),
    ("10%", "transforms_reduction/trainset_10pct/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"),
    ("25%", "transforms_reduction/trainset_25pct/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz"),
    ("0.01 new with correct ckpt2npz script", "transforms_reduction/trainset_0_01pct/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/new_test_transform.npz")
]

for label, path in paths:
    print(f"\n===== {label.upper()} =====")
    try:
        npz = np.load(path, allow_pickle=True)
        print("Keys:", list(npz.keys()))
        for key in npz.keys():
 

            arr = npz[key]
            print(f"  {key}: shape={getattr(arr, 'shape', None)} dtype={arr.dtype if hasattr(arr, 'dtype') else type(arr)}")
            # Print mean/std for W or b
            if key.lower() == "w":
                print(f"    W stats: mean={arr.mean():.5f}, std={arr.std():.5f}, min={arr.min():.5f}, max={arr.max():.5f}")
            if key.lower() == "b":
                print(f"    b stats: mean={arr.mean():.5f}, std={arr.std():.5f}, min={arr.min():.5f}, max={arr.max():.5f}")
    except Exception as e:
        print(f"  Failed to load {path}: {e}")
print()

paths_to_check = [
    "transforms/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/transform.npz",
    "transforms_reduction/trainset_0_01pct/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024/new_test_transform.npz"
]

for path in paths_to_check:
    print(f"\nOriginal weights and biases of the transform at:\n{path}\n")
    try:
        npz = np.load(path, allow_pickle=True)
        print("Keys:", list(npz.keys()))

        # Try 'weights' and 'bias'
        if 'weights' in npz:
            arr = npz['weights']
            print(f"  weights stats: mean={arr.mean():.5f}, std={arr.std():.5f}, min={arr.min():.5f}, max={arr.max():.5f}")
        elif 'W' in npz:
            arr = npz['W']
            print(f"  W stats: mean={arr.mean():.5f}, std={arr.std():.5f}, min={arr.min():.5f}, max={arr.max():.5f}")

        if 'bias' in npz:
            arr = npz['bias']
            print(f"  bias stats: mean={arr.mean():.5f}, std={arr.std():.5f}, min={arr.min():.5f}, max={arr.max():.5f}")
        elif 'b' in npz:
            arr = npz['b']
            print(f"  b stats: mean={arr.mean():.5f}, std={arr.std():.5f}, min={arr.min():.5f}, max={arr.max():.5f}")
    except Exception as e:
        print(f"  Failed to load {path}: {e}")
