import pickle
import pandas as pd

labels = [
    "0.01% (400m)",
    "0.01% (2b)",
    "1% (400m)",
    "1% (2b)",
    "random W",
]

result_paths = [
    "fewshot_results_trainset_0_01pct_400m/cifar100/custom/OpenCLIP_ViT-L-14_laion400m_e32/penultimate/0.001/0.001/0.1/1.0/1024/False/fewshot_results.pkl",
    "fewshot_results_trainset_0_01pct/cifar100/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/0.001/0.001/0.1/1.0/1024/False/fewshot_results.pkl",
    "fewshot_results_trainset_01pct_400m/cifar100/custom/OpenCLIP_ViT-L-14_laion400m_e32/penultimate/0.001/0.001/0.1/1.0/1024/False/fewshot_results.pkl",
    "fewshot_results_trainset_01pct/cifar100/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/0.001/0.001/0.1/1.0/1024/False/fewshot_results.pkl",
    "fewshot_results_randomW_0_01pct_400m/cifar100/custom/OpenCLIP_ViT-L-14_laion400m_e32/penultimate/0.001/0.001/0.1/1.0/1024/False/fewshot_results.pkl",
]

for label, path in zip(labels, result_paths):
    with open(path, "rb") as f:
        df = pickle.load(f)
    if isinstance(df, pd.DataFrame):
        mean_acc = df["accuracy"].mean()
        print(f"{label}")
        print(f"  shape: {df.shape}")
        print(f"  mean accuracy: {mean_acc:.4f}")
    else:
        print(f"{label} is not a DataFrame.")
