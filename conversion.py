import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('triplets_large_final_correctednc_correctedorder.csv')

# Assuming the CSV has columns: 'anchor', 'positive', 'negative'
triplets = df[['anchor', 'positive', 'negative']].values

# Save to .npy file
np.save('triplets.npy', triplets)
