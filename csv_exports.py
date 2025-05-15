import pandas as pd
import os

# Set root paths
base_dir = "features"
datasets = ["cifar100", "cifar100-coarse"]
output_dir = "csv_exports"
os.makedirs(output_dir, exist_ok=True)

for dataset in datasets:
    pkl_path = os.path.join(base_dir, dataset, "results", "probing_results.pkl")
    csv_path = os.path.join(output_dir, f"{dataset}_probing_results.csv")
    
    if os.path.exists(pkl_path):
        df = pd.read_pickle(pkl_path)
        df.to_csv(csv_path, index=False)
        print(f"Exported {dataset} results to {csv_path}")
    else:
        print(f"ould not find: {pkl_path}")
