__all__ = ["GlobalTransform", "GlocalTransform"]

import os
import pickle
from dataclasses import dataclass
from typing import Any
import torch
import numpy as np

Array = np.ndarray
FILE_FORMATS = ["pkl", "npz"]


@dataclass
class GlobalTransform:
    source: str = "custom"
    model_name: str = "clip_ViT-B/16"
    module: str = "penultimate"
    path_to_transform: str = (
        "/home/space/datasets/things/transforms/transforms_without_norm.pkl"
    )
    path_to_features: str = (
        "/home/space/datasets/things/probing/embeddings/features.pkl"
    )

    def __post_init__(self) -> None:
        self._load_transform(self.path_to_transform)
        self._load_features(self.path_to_features)

    def _load_features(self, path_to_features: str) -> None:
        assert os.path.isfile(path_to_features), (
            "\nThe provided path does not point to a file.\nChange path.\n"
        )
        with open(path_to_features, "rb") as f:
            feats = pickle.load(f)
        feats = feats[self.source][self.model_name][self.module]
        # immediately coerce to NumPy
        if torch.is_tensor(feats):
            feats = feats.cpu().numpy()
        self.things_mean = feats.mean()
        self.things_std  = feats.std()

    def _load_transform(self, path_to_transform: str) -> None:
        assert os.path.isfile(path_to_transform), (
            f"\nThe provided path does not point to a valid file: {path_to_transform}\n"
        )
        if path_to_transform.endswith("pkl"):
            with open(path_to_transform, "rb") as f:
                transforms = pickle.load(f)
            self.transform = transforms[self.source][self.model_name][self.module]
        elif path_to_transform.endswith("npz"):
            self.transform = np.load(path_to_transform, allow_pickle=True)
        else:
            raise ValueError(
                f"\nThe provided file does not have a valid format. "
                f"Valid formats are: {FILE_FORMATS}\n"
            )

    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize and apply the learned linear transform to a batch of feature vectors.
        """
        # ----------------------------------------------------------------------------
        # 1) Ensure mean/std are plain NumPy
        # ----------------------------------------------------------------------------
        mean = self.things_mean
        std  = self.things_std

        # ----------------------------------------------------------------------------
        # 2) Normalize
        # ----------------------------------------------------------------------------
        features = (features - mean) / std

        # ----------------------------------------------------------------------------
        # 3) Apply linear map
        # ----------------------------------------------------------------------------
        if isinstance(self.transform, dict) and "weights" in self.transform:
            W = self.transform["weights"]
            b = self.transform.get("bias", None)

            # if they were tensors, convert now
            if torch.is_tensor(W):
                W = W.cpu().numpy().astype(features.dtype)
            if b is not None and torch.is_tensor(b):
                b = b.cpu().numpy().astype(features.dtype)

            features = features @ W
            if b is not None:
                features = features + b

        elif isinstance(self.transform, np.ndarray):
            W = self.transform
            if W.shape[0] != W.shape[1]:
                weights = W[:, :-1]
                bias    = W[:, -1]
                features = features @ weights + bias
            else:
                features = features @ W

        else:
            raise ValueError(f"Unsupported transform format: {type(self.transform)}")

        return features


@dataclass
class GlocalTransform:
    root: str = "/home/space/datasets/things/probing/"
    source: str = "custom"
    model: str = "clip_RN50"
    module: str = "penultimate"
    optim: str = "sgd"
    eta: float = 0.001
    lmbda: float = 1.0
    alpha: float = 0.25
    tau: float = 0.1
    contrastive_batch_size: float = 1024
    adversarial: bool = False

    def __post_init__(self) -> None:
        args = [
            self.root, self.source, self.model, self.module,
            self.optim.lower(), self.eta, self.lmbda,
            self.alpha, self.tau, self.contrastive_batch_size,
        ]
        if self.adversarial:
            args.append("adversarial")
        path = os.path.join(*map(str, args))
        self.transform = self._load_transform(path)

    @staticmethod
    def _load_transform(path_to_transform: str) -> Any:
        fn = os.path.join(path_to_transform, "transform.npz")
        assert os.path.isfile(fn), (
            f"\nThe provided path does not point to a valid file: {fn}\n"
        )
        return np.load(fn, allow_pickle=True)

    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize and apply the learned gLocal linear transform to a batch of feature vectors.
        """
        # ----------------------------------------------------------------------------
        # 1) Ensure mean/std are NumPy
        # ----------------------------------------------------------------------------
        try:
            mean = self.transform["mean"].cpu().numpy()
            std  = self.transform["std"].cpu().numpy()
        except Exception:
            mean = self.transform["mean"]
            std  = self.transform["std"]

        # ----------------------------------------------------------------------------
        # 2) Normalize
        # ----------------------------------------------------------------------------
        features = (features - mean) / std

        # ----------------------------------------------------------------------------
        # 3) Apply weights + optional bias
        # ----------------------------------------------------------------------------
        if "weights" in self.transform:
            W = self.transform["weights"]
            b = self.transform.get("bias", None)

            if torch.is_tensor(W):
                W = W.cpu().numpy().astype(features.dtype)
            if b is not None and torch.is_tensor(b):
                b = b.cpu().numpy().astype(features.dtype)

            features = features @ W
            if b is not None:
                features = features + b

        return features
