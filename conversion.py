# conversion.py
import os
import pandas as pd
import numpy as np

csv_path = 'triplet_dataset/triplets_large_final_correctednc_correctedorder.csv'
df = pd.read_csv(csv_path, sep="\t")


print("Loaded CSV with columns:", df.columns)


try:
    triplets = df[['image1', 'image2', 'image3']].to_numpy(dtype=int)
    print("Extracted triplet array of shape:", triplets.shape)
except Exception as e:
    print("Failed to extract triplets:", e)
    exit()

save_dir = 'data/triplets'
os.makedirs(save_dir, exist_ok=True)


try:
    np.save(os.path.join(save_dir, 'train_90.npy'), triplets)
    np.save(os.path.join(save_dir, 'test_10.npy'), triplets[:int(0.1 * len(triplets))])
    np.save(os.path.join(save_dir, 'correct_triplets.npy'), triplets[:int(0.8 * len(triplets))])
    print("" \
    "Saved all .npy files in:", save_dir)
except Exception as e:
    print("Saving failed:", e)
