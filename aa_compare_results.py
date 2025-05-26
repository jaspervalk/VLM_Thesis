import pickle

paths = [
    "fewshot_results_baseline_cifar100coarse/cifar100-coarse/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/0.001/0.001/None/None/None/False/fewshot_results.pkl",
    "test_fewshot_results_reduction_0_01pct/cifar100-coarse/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/0.001/0.001/0.1/1.0/1024/False/fewshot_results.pkl",
    "fewshot_results_random_transform/cifar100-coarse/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/0.001/0.001/0.1/1.0/1024/False/fewshot_results.pkl",
    "test_fewshot_results_reduction_100pct/cifar100-coarse/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/0.001/0.001/0.1/1.0/1024/False/fewshot_results.pkl"
]

results = []
for path in paths:
    with open(path, "rb") as f:
        results.append(pickle.load(f))

def extract_mean_acc(results):
    for key in ['mean_acc', 'mean_accuracy', 'acc', 'accuracy']:
        if key in results:
            return results[key]
    if isinstance(results, dict):
        for v in results.values():
            if isinstance(v, dict):
                for key in ['mean_acc', 'mean_accuracy', 'acc', 'accuracy']:
                    if key in v:
                        return v[key]
    return None

def get_mean(val):
    # Handles Series, lists, numpy arrays, or scalars
    try:
        import numpy as np
        if hasattr(val, "mean"):
            return float(val.mean())
        else:
            return float(np.mean(val))
    except Exception:
        return float(val)

labels = [
    "baseline",
    "0.01% or 412 triplets",
    "random transform",
    "100% transform"
]

mean_accs = [get_mean(extract_mean_acc(r)) for r in results]
print("Mean Accuracy Comparison: Laion2b on cifar100-coarse\n")
for label, acc in zip(labels, mean_accs):
    print(f"Results {label} mean accuracy: {acc:.3f}")
