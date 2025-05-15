import os
import torch
import numpy as np
from typing import Optional, Any, Tuple
from torchvision import datasets
from torchvision.datasets import CIFAR100, DTD, SUN397, ImageNet

Array = np.ndarray
Tensor = torch.Tensor

class EmbeddedImageNet(ImageNet):
    def __init__(
        self,
        root: str,
        embedding_root: str,
        split: str = "train",
        device: str = "cpu",
        **kwargs: Any
    ) -> None:
        super(EmbeddedImageNet, self).__init__(root=root, split=split, **kwargs)
        self.device = torch.device(device)
        print(f"Embedding root (imagenet) {embedding_root}", self.split)
        self.feature_order = sorted(
            [
                os.path.join(embedding_root, self.split, f.name)
                for f in os.scandir(os.path.join(embedding_root, self.split))
                if f.name.endswith("pt")
            ]
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        _, target = self.samples[index]
        sample = torch.load(self.feature_order[index], map_location=torch.device(self.device))

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


def load_dataset(
    name: str,
    data_dir: str,
    train: bool,
    transform=None,
    embeddings: Optional[np.ndarray] = None,
    embeddings_root: Optional[str] = None,
):
    if name == "cifar100" or name == "cifar100-coarse":
        dataset_class = CIFAR100
        if embeddings is not None:
            # 🛠️ Handle embeddings WITH or WITHOUT ["train"]["test"]
            if isinstance(embeddings, dict) and ("train" in embeddings or "test" in embeddings):
                dataset_class = embed_dataset(dataset_class, embeddings["train" if train else "test"])
            else:
                dataset_class = embed_dataset(dataset_class, embeddings)

        dataset = dataset_class(
            root=data_dir,
            train=train,
            download=True,
            transform=transform,
        )

        # Remap fine labels to coarse for cifar100-coarse
        if name == "cifar100-coarse":
            from downstream.fewshot.cifar import get_cifar100_coarse_map
            coarse_map = get_cifar100_coarse_map()
            dataset.targets = [coarse_map[label] for label in dataset.targets]
            print(f"[DEBUG] Remapped CIFAR100 fine labels to 20 coarse superclasses.")

    elif name.lower() == "dtd":
        dataset_class = DTD
        if embeddings is not None:
            dataset_class = embed_dataset(dataset_class, embeddings)
        dataset = dataset_class(
            root=data_dir,
            split="train" if train else "test",
            download=True,
            transform=transform,
        )

    elif name == "SUN397":
        dataset_class = SUN397
        if embeddings is not None:
            dataset_class = embed_dataset(dataset_class, embeddings)
        dataset = dataset_class(
            root=data_dir,
            download=True,
            transform=transform,
        )
        if train:
            split_file = "Training_01.txt"
        else:
            split_file = "Testing_01.txt"
        with open(os.path.join(dataset.root, split_file)) as f:
            lines = f.read()
        file_names = [l for l in lines.split("\n") if not l == ""]
        dataset._image_files = [
            os.path.join(dataset._data_dir, fn[1:]) for fn in file_names
        ]
        dataset._labels = [
            dataset.class_to_idx["/".join(path.split("/")[2:-1])] for path in file_names
        ]

    elif name == "imagenet":
        if embeddings_root is not None:
            dataset = EmbeddedImageNet(
                root=data_dir,
                embedding_root=embeddings_root,
                split="train" if train else "val",
                transform=transform,
            )
        else:
            dataset = ImageNet(
                root=data_dir,
                split="train" if train else "val",
                transform=transform,
            )

    else:
        raise ValueError(f"\nUnknown dataset: {name}\n")

    print(f"[DEBUG] Loaded {name} with {len(dataset)} samples. Unique labels: {set(dataset.targets)}")
    return dataset



def embed_dataset(dataset, embeddings):
    """Wraps a dataset such that it uses the given embeddings as features."""

    def __getitem__(self, idx):
        if hasattr(self, "targets"):
            label = self.targets[idx]
        else:
            label = self._labels[idx]

        if hasattr(self, "_image_files"):
            embedding = embeddings[str(self._image_files[idx])]
        elif hasattr(self, "indices"):
            embedding = embeddings[self.indices[idx]]
        else:
            embedding = embeddings[idx]

        if self.target_transform:
            label = self.target_transform(label)
        return embedding, label

    print("Dataset embedded.")
    return type(
        dataset.__name__,
        (dataset,),
        {
            "__getitem__": __getitem__,
        },
    )

