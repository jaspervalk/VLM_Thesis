import os
import pandas as pd
import pickle

def detect_transform_from_path(path_parts):
    """Determine if transform is applied based on directory values."""
    try:
        alpha = path_parts[-5]
        tau = path_parts[-4]
        batch_size = path_parts[-3]
        return alpha != "None" and tau != "None" and batch_size != "None"
    except IndexError:
        return False

def collect_results(base_dir="fewshot_results"):
    records = []
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
                    transform = detect_transform_from_path(parts)

                    records.append({
                        "dataset": dataset,
                        "model": model,
                        "transform": transform,
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
        summary = df.groupby(["dataset", "model", "transform"]).agg(mean_acc=("mean_acc", "mean")).reset_index()
        print("\nMean Accuracy Summary\n")
        print(summary.sort_values(["dataset", "model", "transform"]).to_string(index=False))
