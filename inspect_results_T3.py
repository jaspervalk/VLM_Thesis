import numpy as np
import pickle


results = [
    ("random", "fewshot_results/transforms3/random/cifar100/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/0.001/0.001/0.1/1.0/1024/False/fewshot_results.pkl"),
    ("no transform", "fewshot_results/transforms3/no_trans/cifar100/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/0.001/0.001/None/None/None/False/fewshot_results.pkl"),
    ("1pct", "fewshot_results/transforms3/cifar100/1pct/cifar100/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/0.001/0.001/0.1/1.0/1024/False/fewshot_results.pkl"),
    ("0.01pct", "fewshot_results/transforms3/cifar100/0.01pct/cifar100/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/0.001/0.001/0.1/1.0/1024/False/fewshot_results.pkl"),
    ("0.01pct 0.5alpha", "fewshot_results/transforms3/cifar100/0.01pct/alpha0.5/cifar100/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/0.001/0.001/0.1/1.0/1024/False/fewshot_results.pkl")
]

for label, path in results:
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        # Try common keys for accuracy
        if isinstance(data, dict):
            if "acc" in data:
                acc = data["acc"]
            elif "mean_acc" in data:
                acc = data["mean_acc"]
            else:
                acc = None
        else:
            acc = None
        # Handle pandas DataFrame
        if acc is None:
            try:
                import pandas as pd
                if "pandas.core.frame.DataFrame" in str(type(data)):
                    # Try common column names for accuracy
                    for col in ["acc", "accuracy", "mean_acc"]:
                        if col in data.columns:
                            acc = data[col].values
                            break
            except ImportError:
                pass
        if acc is not None:
            if isinstance(acc, (list, np.ndarray)):
                mean_acc = np.mean(acc)
            else:
                mean_acc = acc
            print(f"{label}: mean acc = {mean_acc:.4f}")
        else:
            if isinstance(data, dict):
                print(f"{label}: accuracy not found in file, available keys: {list(data.keys())}")
            elif "pandas.core.frame.DataFrame" in str(type(data)):
                print(f"{label}: accuracy not found in DataFrame, available columns: {list(data.columns)}")
            else:
                print(f"{label}: accuracy not found in file, data type: {type(data)}")
    except Exception as e:
        print(f"{label}: error loading file ({e})")