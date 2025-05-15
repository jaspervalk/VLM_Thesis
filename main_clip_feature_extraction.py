import argparse
import os
from typing import List

import numpy as np
import torch
from thingsvision import get_extractor
from thingsvision.utils.data import DataLoader
from utils.evaluation.helpers import save_features
from torchvision.transforms import Compose, Lambda
from tqdm import tqdm

from data import DATASETS, load_dataset
import json

with open("model_dict.json", "r") as f:
    MODEL_DICT = json.load(f)




def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--data_root", type=str, help="path/to/dataset", default="../human_alignment/datasets")
    aa("--datasets", type=str, nargs="+", help="for which datasets to perform feature extraction", choices=DATASETS)
    aa("--stimulus_set", type=str, default=None, choices=["set1", "set2"], help="Stimulus set for King et al. (2019)")
    aa("--category", type=str, default=None, choices=["animals", "automobiles", "fruits", "furniture", "various", "vegetables"])
    aa("--model_names", type=str, nargs="+", help="models for which to extract features")
    aa("--source", type=str, default="custom", choices=["custom", "timm", "torchvision", "vissl", "ssl", "clip", "openclip"])
    aa("--batch_size", type=int, default=64, help="Mini-batch size")
    aa("--features_root", type=str, default="./features", help="path/to/output/features")
    aa("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="cpu or cuda")
    aa("--extract_cls_token", action="store_true", help="only extract [CLS] token from a ViT model")
    return parser.parse_args()


def load_extractor(model_name: str, source: str, device: str, extract_cls_token: bool = False):
    if model_name.startswith("OpenCLIP"):
        if "laion" in model_name:
            meta_vars = model_name.split("_")
            name = meta_vars[0]
            variant = meta_vars[1]
            data = "_".join(meta_vars[2:])
        else:
            name, variant, data = model_name.split("_")
        model_params = dict(variant=variant, dataset=data)
    elif model_name.startswith("clip"):
        name, variant = model_name.split("_")
        model_params = dict(variant=variant)
    elif model_name.startswith("DreamSim"):
        model_name = model_name.split("_")
        name = model_name[0]
        variant = "_".join(model_name[1:])
        model_params = dict(variant=variant)
    elif extract_cls_token:
        name = model_name
        model_params = dict(extract_cls_token=True)
    else:
        name = model_name
        model_params = None

    return get_extractor(model_name=name, source=source, device=device, pretrained=True, model_parameters=model_params)


def feature_extraction(
    datasets: List[str],
    model_names: List[str],
    source: str,
    device: str,
    batch_size: int,
    data_root: str,
    features_root: str,
    category: str = None,
    stimulus_set: str = None,
    extract_cls_token: bool = False,
) -> None:
    for dataset in tqdm(datasets, desc="Dataset"):
        for model_name in tqdm(model_names, desc="Model"):
            extractor = load_extractor(model_name, source, device, extract_cls_token)
            transformations = extractor.get_transformations()

            if dataset == "peterson":
                assert isinstance(category, str), "\nCategory required for Peterson dataset.\n"
                transformations = Compose([Lambda(lambda img: img.convert("RGB")), transformations])

            data = load_dataset(
                name=dataset,
                data_dir=os.path.join(data_root, dataset),
                stimulus_set=stimulus_set if dataset == "free-arrangement" else None,
                category=category if dataset == "peterson" else None,
                transform=transformations,
                return_label=False,
            )

            batches = DataLoader(dataset=data, batch_size=batch_size, backend=extractor.get_backend())
            module_name = MODEL_DICT[model_name]["penultimate"]["module_name"]
            print(f"Extracting from module: {module_name}")
            features = extractor.extract_features(batches=batches, module_name=module_name, flatten_acts=True)


            # Get output directory and ensure it exists
            out_dir = os.path.join(features_root, dataset, source, model_name, "penultimate")
            os.makedirs(out_dir, exist_ok=True)

            # Include filenames for evaluation script compatibility
            try:
                filenames = np.array([os.path.splitext(os.path.basename(p))[0] for p in data.image_paths])
            except AttributeError:
                filenames = np.array([f"img_{i:05d}" for i in range(len(data))])


            save_features(
                {'penultimate': features, 'filenames': filenames},
                out_path=out_dir,
            )


if __name__ == "__main__":
    args = parseargs()
    feature_extraction(
        datasets=args.datasets,
        model_names=args.model_names,
        source=args.source,
        device=args.device,
        batch_size=args.batch_size,
        data_root=args.data_root,
        features_root=args.features_root,
        category=args.category,
        stimulus_set=args.stimulus_set,
        extract_cls_token=args.extract_cls_token,
    )

