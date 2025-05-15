import os
import pickle

ROOT_DIR = "features"  # Set this to your actual base features folder (e.g., "features/cifar100")
WRAPPED_FEATURES_PATH = "wrapped_features.pkl"  # Output file

def collect_feature_files(root_dir):
    feature_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        if "features.pkl" in filenames:
            feature_files.append(os.path.join(dirpath, "features.pkl"))
    return feature_files

def parse_path(filepath):
    # Example: features/cifar100/custom/openclip_laion2b/penultimate/features.pkl
    parts = filepath.split(os.sep)
    try:
        source = parts[-4]           # e.g., custom
        model = parts[-3]            # e.g., openclip_laion2b
        module = parts[-2]           # e.g., penultimate
        return source, model, module
    except IndexError:
        print(f"⚠️ Could not parse {filepath}")
        return None, None, None

def wrap_features(feature_files):
    features_dict = {}
    for filepath in feature_files:
        source, model, module = parse_path(filepath)
        if None in (source, model, module):
            continue
        with open(filepath, "rb") as f:
            raw = pickle.load(f)
        if source not in features_dict:
            features_dict[source] = {}
        if model not in features_dict[source]:
            features_dict[source][model] = {}
        features_dict[source][model][module] = {
            "features": raw.get("penultimate") if raw.get("penultimate") is not None else raw.get("features"),
            "filenames": raw.get("filenames")
        }
        print(f"✅ Wrapped: {source} / {model} / {module}")
    return features_dict

def main():
    feature_files = collect_feature_files(ROOT_DIR)
    print(f"🔍 Found {len(feature_files)} feature files.")
    features_dict = wrap_features(feature_files)
    with open(WRAPPED_FEATURES_PATH, "wb") as f:
        pickle.dump(features_dict, f)
    print(f"\n💾 Saved wrapped features to: {WRAPPED_FEATURES_PATH}")

if __name__ == "__main__":
    main()
