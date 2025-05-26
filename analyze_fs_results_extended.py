import os
import pandas as pd
import pickle

def detect_transform_from_path(path_parts):
    """
    Detect transform mode from path structure.
    Returns a descriptive label or False.
    """
    if "fewshot_results_reduction_0_01pct" in path_parts:
        return "0.01% triplets"
    if "fewshot_results_reduction_0_1pct" in path_parts:
        return "0.1% triplets"
    if "fewshot_results_reduction_01pct" in path_parts:
        return "1% triplets"
    if "fewshot_results_reduction_10pct" in path_parts:
        return "10% triplets"
    if "fewshot_results_reduction_25pct" in path_parts:
        return "25% triplets"
    if "fewshot_results_reduction_50pct" in path_parts:
        return "50% triplets"
    if "fewshot_results_reduction_75pct" in path_parts:
        return "75% triplets"
    try:
        alpha = path_parts[-5]
        tau = path_parts[-4]
        batch_size = path_parts[-3]
        return "transformed" if alpha != "None" and tau != "None" else "baseline"
    except IndexError:
        return "unknown"

def collect_results(base_dirs=("fewshot_results", "fewshot_results_reduction_0_01pct", "fewshot_results_reduction_01pct", "fewshot_results_reduction_10pct", "fewshot_results_reduction_25pct", "fewshot_results_reduction_50pct", "fewshot_results_reduction_75pct")):
    records = []
    for base_dir in base_dirs:
        for root, dirs, files in os.walk(base_dir):
            if "fewshot_results.pkl" in files:
                full_path = os.path.join(root, "fewshot_results.pkl")
                try:
                    with open(full_path, "rb") as f:
                        data = pickle.load(f)
                        df = pd.DataFrame(data)

                        if "accuracy" not in df.columns:
                            print(f"[WARN] Skipping {full_path} — missing 'accuracy' column.")
                            continue

                        parts = full_path.split(os.sep)
                        dataset = parts[1]
                        model = parts[3]
                        transform_label = detect_transform_from_path(parts)

                        records.append({
                            "dataset": dataset,
                            "model": model,
                            "transform": transform_label,
                            "mean_acc": df["accuracy"].mean(),
                            "path": full_path,
                        })
                except Exception as e:
                    print(f"[ERR ] Failed reading {full_path}: {e}")
    return pd.DataFrame(records)

if __name__ == "__main__":
    df = collect_results()
    if df.empty:
        print("No valid results found.")
    else:
        summary = df.groupby(["dataset", "model", "transform"]).agg(
            mean_acc=("mean_acc", "mean")).reset_index()
        print("\nMean Accuracy Summary\n")
        print(summary.sort_values(["dataset", "model", "transform"]).to_string(index=False))
