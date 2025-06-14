import argparse
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from ml_collections import config_dict
from thingsvision import get_extractor

import utils
from downstream.fewshot.breeds_sets import get_breeds_task
from downstream.fewshot.cifar import get_cifar100_coarse_map
from downstream.fewshot.data import load_dataset
from downstream.fewshot.predictors import get_regressor, test_regression
from main_glocal_probing_efficient import get_combination
from main_model_sim_eval import get_module_names
from utils.evaluation.transforms import GlobalTransform, GlocalTransform
from utils.probing.helpers import model_name_to_thingsvision


print("[DEBUG] Current working directory:", os.getcwd())

Array = np.ndarray
Tensor = torch.Tensor
FrozenDict = Any

BREEDS_TASKS = ("living17", "entity13", "entity30", "nonliving26")

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)



def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    # Base arguments
    aa("--data_root", type=str, help="path/to/things")
    aa("--dataset", type=str, help="Which dataset to use", default="imagenet")
    aa(
        "--task",
        type=str,
        choices=["none", "coarse"] + list(BREEDS_TASKS),
        help="Which task to do",
        default="none",
    )
    aa(
        "--model_names",
        type=str,
        nargs="+",
        help="models for which we want to extract features",
    )
    aa(
        "--module",
        type=str,
        choices=["logits", "penultimate"],
        help="module for which to extract features",
    )
    aa("--overall_source", type=str, default="thingsvision")
    aa(
        "--sources",
        type=str,
        nargs="+",
        choices=[
            "custom",
            "torchvision",
            "ssl",
        ],
        help="Source of (pretrained) models",
    )
    aa(
        "--model_dict_path",
        type=str,
        default="/home/space/datasets/things/model_dict.json",
        help="Path to the model_dict.json",
    )
    aa(
        "--input_dim",
        type=int,
        help="Side-length of the input images.",
        default=32,
    )
    # Few shot arguments
    aa(
        "--n_shot",
        type=int,
        nargs="+",
        help="Number samples per class for training",
        default=5,
    )
    aa(
        "--n_test",
        type=int,
        help="Number samples per class for testing",
        default=100,
    )
    aa(
        "--n_reps",
        type=int,
        help="Number of repetitions per experiment",
        default=1,
    )
    aa(
        "--regressor_type",
        type=str,
        nargs="+",
        choices=["ridge", "knn", "tip"],
        default=["ridge"],
        help="Few shot model.",
    )
    aa(
        "--n_classes",
        type=int,
        help="Number of classes",
    )
    aa(
        "--class_id_set",
        type=int,
        nargs="+",
        help="Classes to use",
        default=None,
    )
    aa(
        "--resample_testset",
        action="store_true",
        help="Whether to re-sample the test samples for each repetition. Should be True if not all test samples are to be used in each iter",
    )
    aa(
        "--sample_per_superclass",
        action="store_true",
        help="Whether to sample the shots for each superclass, rather than each class.",
    )
    aa(
        "--solver",
        type=str,
        default="lbfgs",
        help="Solver to use for ridge regression",
        choices=["lbfgs", "sag"],
    )
    # Transform arguments
    aa("--optim", type=str, default="SGD", choices=["Adam", "AdamW", "SGD"])
    aa(
        "--etas",
        type=float,
        default=1e-3,
        nargs="+",
    )
    aa(
        "--lmbdas",
        type=float,
        default=1e-3,
        nargs="+",
        help="Relative contribution of the l2 or identity regularization penality",
    )
    aa(
        "--alphas",
        type=float,
        default=1e-1,
        nargs="+",
        help="Relative contribution of the contrastive loss term",
    )
    aa(
        "--taus",
        type=float,
        default=1,
        nargs="+",
        help="temperature value for contrastive learning objective",
    )
    aa(
        "--contrastive_batch_sizes",
        type=int,
        default=1024,
        nargs="+",
        metavar="B_C",
    )
    aa(
        "--transform_type",
        type=str,
        default="glocal",
        choices=["glocal", "global", "naive", "naive_bias", "without"],
    )
    # Misc arguments
    aa("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    aa(
        "--things_embeddings_path",
        type=str,
        default="/home/space/datasets/things/embeddings/model_features_per_source.pkl",
        help="path/to/things/embeddings/file",
    )
    aa(
        "--transforms_root",
        type=str,
        default="/home/space/datasets/things/probing/results",
        help="path/to/embeddings",
    )
    aa(
        "--embeddings_root",
        type=str,
        default=None,
        help="path/to/embeddings of the dataset",
    )
    aa(
        "--zero_shot_root",
        type=str,
        default=None,
        help="path/to/zero_shot_weights. only reqired for tip-adapter.",
    )
    aa(
        "--full_data",
        action="store_true",
        help="Whether to use the transformed trained on the full data.",
    )
    aa(
        "--adversarial",
        action="store_true",
        help="Whether to use the adversarial transforms.",
    )
    aa("--out_dir", type=str, help="directory to save the results to")
    aa("--rnd_seed", type=int, default=42, help="random seed for reproducibility")
    args = parser.parse_args()
    return args


def is_embedding_source(source: str) -> bool:
    return source not in ["torchvision", "custom", "ssl"]


def get_subset_indices(dataset: Any, cls_id: Union[int, List[int]]) -> List[int]:
    if isinstance(cls_id, int):
        cls_id = [cls_id]
    attr = "targets" if hasattr(dataset, "targets") else "_labels"
    subset_indices = [
        i_cls for i_cls, cls in enumerate(getattr(dataset, attr)) if cls in cls_id
    ]
    return subset_indices


def get_features_targets(
    class_ids,
    model_name,
    model_params,
    source,
    module,
    module_type,
    data_cfg,
    batch_size,
    train,
    ids_subset=None,
    n_batches=1,
    shuffle=False,
    device: str = "cpu",
    embeddings: Optional[np.ndarray] = None,
    superclass_mapping: Optional[Dict] = None,
    sample_per_superclass: bool = False,
):
    ids_subset = class_ids if ids_subset is None else ids_subset
    dataset_is_embedded = is_embedding_source(source) or embeddings is not None

    if dataset_is_embedded:
        if data_cfg.name == "dtd":
            print("\n[DEBUG] --- DTD LOAD_DATASET CALL ---")
            print(f"Calling load_dataset with: name={data_cfg.name}, data_dir={data_cfg.root}, train={train}")
        print(f"Using pre-loaded embeddings for {model_name}")

        dataset = load_dataset(
            name=data_cfg.name,
            data_dir=data_cfg.root,
            train=train,
            embeddings=embeddings,
        )
        if data_cfg.name == "dtd":
            print(f"[DEBUG] dataset.root after loading: {getattr(dataset, 'root', None)}")

    else:
        complete_model_name = model_name
        if model_params is not None:
            variant = model_params.get("variant")
            dataset = model_params.get("dataset")
            if variant:
                complete_model_name += f"_{variant}"
            if dataset and dataset not in complete_model_name:
                complete_model_name += f"_{dataset}"

        print(f"Computed model path: {complete_model_name}")


        embedding_base = os.path.join(data_cfg.embeddings_root)
        if source not in embedding_base:
            embedding_base = os.path.join(embedding_base, source)

        embeddings_path = os.path.join(
            embedding_base,
            complete_model_name,
            module_type,
        )

        print(f"Final embeddings path: {embeddings_path}")

        print(f"Trying to load embeddings from: {embeddings_path}")
        try:
            with open(os.path.join(embeddings_path, "embeddings.pkl"), "rb") as f:
                embeddings = pickle.load(f)
                dataset_is_embedded = True
                print("Loaded embeddings from disk.")
        except FileNotFoundError:
            print("Embeddings not found on disk. Falling back to extractor.")

        if dataset_is_embedded:
            dataset = load_dataset(
                name=data_cfg.name,
                data_dir=data_cfg.root,
                train=train,
                embeddings=embeddings,
                embeddings_root=embeddings_path,
            )
        else:
            extractor = get_extractor(
                model_name=model_name,
                source=source,
                device=device,
                pretrained=True,
                model_parameters=model_params,
            )
            transform = extractor.get_transformations()
                    
            if data_cfg.name.startswith("cifar") and args.input_dim == 224:
                from torchvision import transforms
                print("Resizing CIFAR images to 224x224 for ViT compatibility.")
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transform
                ])

            print(f"Final transform pipeline: {transform}")
            dataset = load_dataset(
                name=data_cfg.name,
                data_dir=data_cfg.root,
                train=train,
                transform=transform, 
            )


    if sample_per_superclass:
        if superclass_mapping is None:
            raise ValueError("Superclass mapping required for sampling per superclass.")
        n_superclasses = len(set(superclass_mapping.values()))
        class_ids = [
            [cls_id for cls_id in ids_subset if superclass_mapping[cls_id] == i]
            for i in range(n_superclasses)
        ]


    features_all = []
    Y_all = []
    for i_batch in range(n_batches):
        indices = []
        for cls_id in class_ids:
            if isinstance(cls_id, int) and cls_id not in ids_subset:
                continue
            subset_indices = get_subset_indices(dataset, cls_id)
            indices.extend(
                list(np.random.choice(subset_indices, size=batch_size, replace=False))
            )
    
        print(f"[DEBUG] Sampled indices from test set: {len(indices)} for class_ids: {class_ids}")


        subset = torch.utils.data.Subset(dataset, indices)
        batches = torch.utils.data.DataLoader(
            subset,
            batch_size=len(indices),
            shuffle=shuffle,
            num_workers=0,
            worker_init_fn=seed_worker,
        )
        X, Y = next(iter(batches))
        X = X.to(device)

        if len(Y.shape) > 1 and Y.shape[1] > 1:
            Y = torch.argmax(Y, dim=1)
        if superclass_mapping is not None:
            Y = [superclass_mapping[int(y)] for y in Y]
        Y = np.array(Y)

        print(f"[DEBUG] Y shape: {Y.shape}, Unique Y values: {np.unique(Y)}")

        if dataset_is_embedded:
            features = X.detach().cpu().numpy()
        else:
            features = extractor.extract_features(
                batches=X,
                module_name=module,
                flatten_acts=True,
            )
            if i_batch == 0:  # Only save first batch (e.g., full CIFAR train or test set)
                save_path = os.path.join(
                    "features",
                    data_cfg.name,
                    source,
                    complete_model_name,
                    module_type,
                    "train" if train else "test"
                )
                os.makedirs(save_path, exist_ok=True)

                save_file = os.path.join(save_path, "embeddings.pkl")
                print(f"Saving extracted embeddings to: {save_file}")
                with open(save_file, "wb") as f:
                    pickle.dump({idx: feat for idx, feat in zip(indices, features)}, f)

        features_all.append(features)
        Y_all.append(Y)

    return features_all, Y_all



def create_config_dicts(args, embedding_keys=None) -> Tuple[FrozenDict, FrozenDict]:
    """Create data and model config dictionaries."""
    model_config = utils.evaluation.load_model_config(args.model_dict_path)
    model_cfg = config_dict.ConfigDict()
    data_cfg = config_dict.ConfigDict()
    model_cfg.module_type = args.module
    model_cfg.sources = args.sources
    model_cfg.input_dim = args.input_dim
    if embedding_keys is not None:
        model_cfg.embeddings_root = args.embeddings_root  # .split("/")[-1]
        model_cfg.names = [k for k in embedding_keys]
    else:
        embeddings_root = (
            args.embeddings_root if hasattr(args, "embeddings_root") else None
        )
        model_cfg.embeddings_root = embeddings_root
        model_cfg.names = args.model_names
    data_cfg.embeddings_root = model_cfg.embeddings_root
    model_cfg.modules = get_module_names(model_config, model_cfg.names, args.module)
    model_cfg = config_dict.FrozenConfigDict(model_cfg)
    data_cfg.root = args.data_root
    data_cfg.name = args.dataset
    data_cfg.resample_testset = args.resample_testset
    data_cfg.category = args.category if hasattr(args, "category") else None
    data_cfg = config_dict.FrozenConfigDict(data_cfg)
    return model_cfg, data_cfg


def run(
    n_shot: int,
    n_test: int,
    n_reps: int,
    class_id_set: List,
    device: str,
    model_cfg: FrozenDict,
    data_cfg: FrozenDict,
    transforms: Dict,
    regressor_type: str = "ridge",
    class_id_set_test: Optional[List] = None,
    superclass_mapping: Optional[Dict] = None,
    sample_per_superclass: bool = False,
    model_id_in_cfg: int = 0,
    embeddings: Optional[Dict] = None,
    solver: str = "lbfgs",
    transform: bool = True,
    zero_shot_weights: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    import hashlib


    if class_id_set_test is None:
        class_id_set_test = class_id_set
        print("Using training classes for testing")

    model_name, module, source = (
        model_cfg.names[model_id_in_cfg],
        model_cfg.modules[model_id_in_cfg],
        model_cfg.sources[model_id_in_cfg],
    )

    # Resolve family name
    name, model_params = model_name_to_thingsvision(model_name)
    family_name = utils.analyses.get_family_name(model_name)
    module_type = model_cfg.module_type

    if embeddings is not None:
        embeddings = embeddings[model_name]

    # Extract train features
    start_t_train_data = datetime.now()
    train_features_all, train_targets_all = get_features_targets(
        class_ids=class_id_set,
        model_name=name,
        model_params=model_params,
        source=source,
        module=module,
        module_type=module_type,
        data_cfg=data_cfg,
        batch_size=n_shot,
        train=True,
        n_batches=n_reps,
        shuffle=True,
        device=device,
        superclass_mapping=superclass_mapping,
        sample_per_superclass=sample_per_superclass,
        embeddings=embeddings,
    )
    end_t_train_data = datetime.now()
    print("Time to load train data: ", (end_t_train_data - start_t_train_data))

    if transform and zero_shot_weights is not None:
        # Align text features
        zero_shot_weights = transforms[source][model_name].transform_features(
            zero_shot_weights
        )

    # Fit multinomial logitstic regression
    regressors = []
    for train_features, train_targets in zip(train_features_all, train_targets_all):
        # This loops over the repetitions
        if transform:
            transform_dict = transforms[source][model_name].transform
            if "weights" in transform_dict:
                W = transform_dict["weights"]
                b = transform_dict.get("bias", None)
            elif "W" in transform_dict:
                W = transform_dict["W"]
                b = transform_dict.get("b", None)
            else:
                raise KeyError("No weights or W key in transform dict")

            print("[DEBUG] W hash:", hashlib.md5(W.tobytes()).hexdigest())
            train_features = transforms[source][model_name].transform_features(train_features)
            train_features = np.nan_to_num(train_features, nan=0.0, posinf=0.0, neginf=0.0)
            import hashlib

            print("[DEBUG] Loaded W for FS eval:")
            print("  shape:", W.shape)
            print("  norm:", np.linalg.norm(W))
            print("  sum:", np.sum(W))
            print("  hash:", hashlib.md5(W.astype(np.float32).tobytes()).hexdigest())


        regressor = get_regressor(
            train_features=train_features,
            train_targets=train_targets,
            regressor_type=regressor_type,
            k=n_shot,
            solver=solver,
            zero_shot_weights=zero_shot_weights
        )
        regressors.append(regressor)

    # Extract and evaluate features for each class individually.
    results = []
    for i_rep in range(n_reps):
        if i_rep == 0 or data_cfg.resample_testset:
            start_t_train_data = datetime.now()
            test_features, test_targets = get_features_targets(
                class_ids=class_id_set_test,
                model_name=name,
                model_params=model_params,
                source=source,
                module=module,
                module_type=module_type,
                data_cfg=data_cfg,
                batch_size=n_test,
                train=False,
                device=device,
                superclass_mapping=superclass_mapping,
                embeddings=embeddings,
            ) 
            test_features = test_features[0]
            test_targets = test_targets[0]
            end_t_train_data = datetime.now()
            print("Time to load test data: ", (end_t_train_data - start_t_train_data))

            if transform:
                print("\n[DEBUG] Applying transform for test features")
                print(f"[DEBUG] Model: {model_name}")
                print(f"[DEBUG] Source: {source}")
                print(f"[DEBUG] Transform class: {type(transforms[source][model_name]).__name__}")

                if hasattr(transforms[source][model_name], "root"):
                    print(f"[DEBUG] Transform root: {transforms[source][model_name].root}")
                if hasattr(transforms[source][model_name], "transform") and isinstance(transforms[source][model_name].transform, dict):
                    transform_dict = transforms[source][model_name].transform
                    W = transform_dict.get("weights", transform_dict.get("W", None))
                    if W is not None:
                        print(f"[DEBUG] W shape: {W.shape} | W sum: {W.sum():.4f}")


                # Print feature stats before and after transform (add these!)
                print("[DEBUG] Test features before transform:")
                print("  mean:", np.mean(test_features), "std:", np.std(test_features), "norm:", np.linalg.norm(test_features))

                # Apply transform
                test_features = transforms[source][model_name].transform_features(test_features)
                test_features = np.nan_to_num(test_features, nan=0.0, posinf=0.0, neginf=0.0)

                print("[DEBUG] Test features AFTER transform:")
                print("  mean:", np.mean(test_features), "std:", np.std(test_features), "norm:", np.linalg.norm(test_features))
                print("[DEBUG] First 10 elements of first test feature:", test_features[0][:10])



        acc, _ = test_regression(
            regressors[i_rep],
            test_targets,
            test_features,
        )
        # save results for all classes
        summary = {
            "accuracy": acc,
            "model": model_name,
            "module": model_cfg.module_type,
            "source": source,
            "family": family_name,
            "dataset": data_cfg.name,
            "transform": transform,
            "classes": list(set(class_id_set).union(set(class_id_set_test))),
            "n_train": n_shot,
            "repetition": i_rep,
            "regressor": regressor_type,
            "samples_per_superclass": sample_per_superclass,
        }
        summary.update(
            {
                att: getattribute(transforms[source][model_name], att)
                for att in [
                    "optim",
                    "eta",
                    "lmbda",
                    "alpha",
                    "tau",
                    "contrastive_batch_size",
                ]
            }
        )
        results.append(summary)
    results = pd.DataFrame(results)
    return results


def getattribute(object: object, att: str) -> Union[bool, float, int, str]:
    if hasattr(object, att):
        return getattr(object, att)
    return None


if __name__ == "__main__":
    start_t = datetime.now()

    # parse arguments
    args = parseargs()
    original_dataset_name = args.dataset
    # if args.dataset == "cifar100-coarse":
    #     args.dataset = "cifar100"



    if args.task in BREEDS_TASKS:
        # Breeds tasks already return fine label IDs and a mapping
        class_id_set, class_id_set_test, superclass_mapping = get_breeds_task(args.task)

    elif args.dataset in {"cifar100", "cifar100-coarse"} and args.task == "coarse":
        # Use CIFAR100 fine labels (0-99), and map them to 20 coarse labels
        from downstream.fewshot.cifar import get_cifar100_coarse_map
        superclass_mapping = get_cifar100_coarse_map()
        class_id_set = class_id_set_test = [i for i in range(100)]  # Fine label IDs

    else:
        # Default fallback: no coarse task, no mapping
        args.task = None
        superclass_mapping = None
        if args.class_id_set is None:
            class_id_set = [i for i in range(args.n_classes)]
        else:
            class_id_set = args.class_id_set
        class_id_set_test = class_id_set

    n_test = args.n_test
    device = torch.device(args.device)


    # Load embeddings for all models
    if args.embeddings_root is not None:
        try:
            embeddings = utils.evaluation.load_embeddings(
                embeddings_root=args.embeddings_root,
                module="embeddings" if args.module == "penultimate" else "logits",
            )
        except:
            print("Could not load embeddings. Continuing without embeddings.")
            embeddings = None
        if isinstance(embeddings, dict):
            embeddings = embeddings[
                "embeddings" if args.module == "penultimate" else "logits"
            ]
    else:
        embeddings = None
    model_cfg, data_cfg = create_config_dicts(args, None)

    # Prepare for loading transforms
    transforms = {
        source: {model_name: {} for model_name in model_cfg.names}
        for source in model_cfg.sources
    }
    if args.transform_type != "glocal":
        args.alphas = [None]
        args.taus = [None]
        args.contrastive_batch_sizes = [None]
    eta, lmbda, alpha, tau, contrastive_batch_size = get_combination(
        etas=args.etas if isinstance(args.etas, list) else [args.etas],
        lambdas=args.lmbdas if isinstance(args.lmbdas, list) else [args.lmbdas],
        alphas=args.alphas if isinstance(args.alphas, list) else [args.alphas],
        taus=args.taus if isinstance(args.taus, list) else [args.taus],
        contrastive_batch_sizes=args.contrastive_batch_sizes if isinstance(args.contrastive_batch_sizes, list) else [args.contrastive_batch_sizes],
)

    all_results = []
    regressor_types = args.regressor_type
    n_shots = args.n_shot
    for model_id_in_cfg, (src, model_name, module) in enumerate(
        zip(model_cfg.sources, model_cfg.names, model_cfg.modules)
    ):
        # Create out path and check if results already exist
        out_path = os.path.join(
            args.out_dir,
            args.dataset + ("" if args.task is None else f"_{args.task}"),
            model_cfg.sources[model_id_in_cfg],
            model_cfg.names[model_id_in_cfg],
            model_cfg.module_type,
            str(eta),
            str(lmbda),
            str(alpha),
            str(tau),
            str(contrastive_batch_size),
            str(args.sample_per_superclass),
        )
        if not os.path.exists(out_path):
            print("\nOutput directory does not exist...")
            print("Creating output directory to save results...\n")
            os.makedirs(out_path)
        out_file_path = os.path.join(out_path, "fewshot_results.pkl")

        if os.path.isfile(out_file_path):
            print("Results already exist. Skipping...")
            print(f"Results file: {out_file_path}")
            continue

        print("Transform root: ", args.transforms_root)
        if args.transform_type == "glocal":
            try:
                optimal = pd.read_pickle(
                    os.path.join(args.transforms_root, "optimally_aligned_probes.pkl")
                )
                is_opt = (
                    len(
                        optimal[
                            (optimal["model"] == model_name)
                            & (optimal["lr"] == eta)
                            & (optimal["lmbda"] == lmbda)
                            & (optimal["alpha"] == alpha)
                            & (optimal["tau"] == tau)
                            & (
                                optimal["contrastive_batch_size"]
                                == contrastive_batch_size
                            )
                        ]
                    )
                    > 0
                )
                if is_opt == 0:
                    print("Transforms not optimal. Skipping...")
                    continue
            except FileNotFoundError:
                print(
                    "Could not load optimal transforms. Continuing without checking for optimality."
                )

        # Load transforms
        try:
            if args.transform_type == "without":
                transforms[src][model_name] = None
            elif args.transform_type != "glocal":
                try:
                    if args.transform_type == "naive":
                        path_to_transform = os.path.join(
                            args.transforms_root, "naive_transforms.pkl"
                        )
                    elif args.transform_type == "naive_bias":
                        path_to_transform = os.path.join(
                            args.transforms_root,
                            "naive_transforms_full_data.pkl"
                            if args.full_data
                            else "naive_transforms_plus_bias.pkl",
                        )
                    else:
                        path_to_transform = os.path.join(
                            args.transforms_root,
                            src,
                            model_name,
                            model_cfg.module_type,
                            "3",
                            str(lmbda),
                            args.optim.lower(),
                            str(eta),
                            "transform.npz",
                        )

                    transforms[src][model_name] = GlobalTransform(
                        source=src,
                        model_name=model_name,
                        module=model_cfg.module_type,
                        path_to_transform=path_to_transform,
                        path_to_features=args.things_embeddings_path,
                    )
                except:
                    # TODO: remove this branch; is just for backward compatibility
                    path_to_transform = os.path.join(
                        args.transforms_root,
                        model_name,
                        model_cfg.module_type,
                        "3",
                        str(lmbda),
                        args.optim.lower(),
                        str(eta),
                        "transform.npz",
                    )
                    transforms[src][model_name] = GlobalTransform(
                        source=src,
                        model_name=model_name,
                        module=model_cfg.module_type,
                        path_to_transform=path_to_transform,
                        path_to_features=args.things_embeddings_path,
                    )
            else:
                # Construct transform path first
                transform_root = os.path.join(args.transforms_root, "full") if args.full_data else args.transforms_root
                transform_path = os.path.join(
                    transform_root,
                    src,
                    model_name,
                    model_cfg.module_type,
                    args.optim.lower(),
                    str(eta),
                    str(lmbda),
                    str(alpha),
                    str(tau),
                    str(contrastive_batch_size),
                    "transform.npz",
                )

                if not os.path.exists(transform_path):
                    print(f"\n Transform not found at: {transform_path}")
                    print("Skipping...\n")
                    continue

                # Load Glocal transform
                transforms[src][model_name] = GlocalTransform(
                    root=transform_root,
                    source=src,
                    model=model_name,
                    module=model_cfg.module_type,
                    optim=args.optim.lower(),
                    eta=eta,
                    lmbda=lmbda,
                    alpha=alpha,
                    tau=tau,
                    contrastive_batch_size=contrastive_batch_size,
                    adversarial=args.adversarial,
                )
                print("Adversarial: ", args.adversarial)

                if "mean" not in transforms[src][model_name].transform.keys():
                    # Backward compatibility with old transforms that don't have mean and std
                    with open(args.things_embeddings_path, "rb") as f:
                        things_features = pickle.load(f)
                    things_features = things_features[src][model_name][
                        model_cfg.module_type
                    ]
                    transforms[src][model_name].transform = dict(
                        transforms[src][model_name].transform
                    )
                    transforms[src][model_name].transform[
                        "mean"
                    ] = things_features.mean()
                    transforms[src][model_name].transform["std"] = things_features.std()
        except AssertionError as e:
            print(e)
            print("Skipping...")
            continue

        # Do few-shot
        print(f"Regressor(s) selected: {args.regressor_type}")
        for regressor_type in regressor_types:
            # Load a zero-shot model, using Tip-Adapter
            if regressor_type == "tip":
                print("Loading zero-shot model from: ", args.zero_shot_root)
                with open(os.path.join(args.zero_shot_root, model_name.replace("/", "-") + ".pkl"), "rb") as f:
                    if args.dataset == "imagenet":
                        # Breeds subsets
                        key = args.task
                    else:
                        key = args.dataset
                        if (args.dataset == "cifar100" and args.task == "coarse"):
                            key += "c"
                    zero_shot_weights = pickle.load(f)[key]
            else:
                zero_shot_weights = None

            n_shots = args.n_shot if isinstance(args.n_shot, list) else [args.n_shot]
            for shots in n_shots:
                if regressor_type == "ridge" and shots == 1:
                    continue
                args.n_shot = shots
                args.regressor_type = regressor_type
                model_cfg, data_cfg = create_config_dicts(args, None)

                np.random.seed(int(1e5))
                torch.manual_seed(int(1e5))

                results = run(
                    n_shot=shots,
                    n_test=args.n_test,
                    n_reps=args.n_reps,
                    class_id_set=class_id_set,
                    class_id_set_test=class_id_set_test,
                    device=args.device,
                    model_cfg=model_cfg,
                    data_cfg=data_cfg,
                    transforms=transforms,
                    regressor_type=args.regressor_type,
                    superclass_mapping=superclass_mapping,
                    sample_per_superclass=args.sample_per_superclass,
                    model_id_in_cfg=model_id_in_cfg,
                    embeddings=embeddings,
                    solver=args.solver,
                    transform=False if args.transform_type == "without" else True,
                    zero_shot_weights=zero_shot_weights,
)
                all_results.append(results)
        results = pd.concat(all_results)
        results["lmbda"] = lmbda
        results["eta"] = eta
        results["optim"] = args.optim.lower()
        print(f"[DEBUG] Saving results to: {out_file_path}")

        results.to_pickle(out_file_path)

    print("Elapsed time (init):", datetime.now() - start_t)
