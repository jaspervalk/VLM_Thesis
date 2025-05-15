import argparse
import numpy as np
import os

def subsample_triplets(input_path, output_path, fraction, seed=42):
    np.random.seed(seed)

    # Load the triplets
    triplets = np.load(input_path)
    print(f"Original triplet count: {len(triplets)}")

    # Compute how many to sample
    n_to_sample = int(len(triplets) * fraction)
    print(f"Subsampling {n_to_sample} triplets ({fraction * 100:.0f}%)")

    # Randomly select a subset
    sampled_indices = np.random.choice(len(triplets), n_to_sample, replace=False)
    subsampled = triplets[sampled_indices]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the result
    np.save(output_path, subsampled)
    print(f"Subsampled triplets saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to original triplet .npy file")
    parser.add_argument("--output", type=str, required=True, help="Where to save the subsampled triplets")
    parser.add_argument("--fraction", type=float, required=True, help="Fraction of triplets to keep (e.g. 0.25 or 0.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    subsample_triplets(args.input, args.output, args.fraction, args.seed)
