import glob
import pickle
import re
import pandas as pd

files = glob.glob("fewshot_GS_400m/**/fewshot_results.pkl", recursive=True)
results = []

for f in sorted(files):
    # Extract alpha from path (same as before)
    m = re.search(r'/([01](?:\.\d+)?)/1\.0/1024/False/fewshot_results\.pkl', f)
    if m:
        alpha = float(m.group(1))
    else:
        m2 = re.search(r'alpha_(\d+\.\d+|\d+)/', f)
        alpha = float(m2.group(1)) if m2 else None

    # Load the result
    with open(f, "rb") as handle:
        d = pickle.load(handle)

    mean_acc = None
    # Try as DataFrame
    if isinstance(d, pd.DataFrame):
        if "accuracy" in d.columns:
            mean_acc = d["accuracy"].mean()
        else:
            print(f"{f} DataFrame but no 'accuracy' column!")
    # Try as dict/list fallback
    elif isinstance(d, dict) and 'accs' in d:
        acc = d['accs']
        mean_acc = float(acc) if not hasattr(acc, '__len__') else float(sum(acc)/len(acc))
    elif isinstance(d, list) and len(d) > 0:
        mean_acc = float(sum(d)/len(d))
    else:
        print(f"{f} could not read mean accuracy!")

    print(f"{f}  alpha={alpha}  mean_acc={mean_acc}")
    results.append((alpha, mean_acc))

# Summary
print("\nSummary (alpha, mean_acc):")
for alpha, acc in sorted(results):
    print(f"alpha={alpha:<4} mean_acc={acc}")
