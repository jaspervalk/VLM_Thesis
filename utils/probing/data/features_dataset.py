import os

import h5py
import torch


class FeaturesPT(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str = "train", device: str = "cuda") -> None:
        super(FeaturesPT, self).__init__()
        self.root = root
        self.split = split
        self.device = torch.device(device)
        loaded = torch.load(os.path.join(self.root, self.split, "features.pt"), map_location="cpu")
        self.data = loaded.to(self.device) if self.device.type != "cpu" else loaded

        if not isinstance(self.data, torch.Tensor):
            raise ValueError("Expected features.pt to be a torch.Tensor.")
        print(f"[DEBUG] Loaded {split} features with shape: {self.data.shape}")

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)



class FeaturesHDF5(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str = "train") -> None:
        super(FeaturesHDF5, self).__init__()
        self.root = root
        self.split = split
        self.h5py_view = h5py.File(
            os.path.join(self.root, self.split, "features.hdf5"), "r"
        )
        self.h5py_key = list(self.h5py_view.keys()).pop()
        # features = torch.from_numpy(self.h5py_view[self.h5py_key][:])
        # self.features = features.to(torch.float32)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.h5py_view[self.h5py_key][idx]).to(torch.float32)
        # return self.features[idx]

    def __len__(self) -> int:
        return self.h5py_view[self.h5py_key].shape[0] 
