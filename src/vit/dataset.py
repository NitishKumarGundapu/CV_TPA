from collections import defaultdict
import csv
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from torchvision.datasets import Flowers102
from transformers import AutoImageProcessor
from PIL import Image


class Flowers102FewShot(Dataset):
    """Oxford Flowers102 dataset formatted for ``ViTClassifier``.

    The training split has 10 examples per class by default. Set
    ``shots_per_class`` to another positive value to sample a different
    number of training examples per class. Validation and test always use
    their complete official splits.
    """

    _SPLITS = {"train": "train", "val": "val", "validation": "val", "test": "test"}

    def __init__(
        self,
        root: str,
        split: str = "train",
        shots_per_class: Optional[int] = 10,
        seed: int = 42,
        model_name: str = "google/vit-base-patch16-224",
        image_processor=None,
        download: bool = True,
    ):
        if split not in self._SPLITS:
            raise ValueError(f"split must be one of {tuple(self._SPLITS)}, got {split!r}")
        if shots_per_class is not None and shots_per_class < 1:
            raise ValueError("shots_per_class must be a positive integer or None")

        self.dataset = Flowers102(
            root=root,
            split=self._SPLITS[split],
            download=download,
        )
        self.image_processor = image_processor or AutoImageProcessor.from_pretrained(model_name)
        self.indices = list(range(len(self.dataset)))

        if shots_per_class is not None and self._SPLITS[split] == "train":
            # Torchvision stores the zero-based class labels for this split.
            labels = getattr(self.dataset, "_labels", None)
            if labels is None:
                labels = [self.dataset[index][1] for index in self.indices]

            by_class = defaultdict(list)
            for index, label in enumerate(labels):
                by_class[int(label)].append(index)

            generator = torch.Generator().manual_seed(seed)
            selected = []
            for label_indices in by_class.values():
                order = torch.randperm(len(label_indices), generator=generator).tolist()
                selected.extend(
                    label_indices[position]
                    for position in order[:shots_per_class]
                )
            self.indices = sorted(selected)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image, label = self.dataset[self.indices[index]]
        image = image.convert("RGB")
        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label, dtype=torch.long),
        }


if __name__ == "__main__":
    dataset = Flowers102FewShot(root="data/Flowers102", split="train", shots_per_class=5)
    print(len(dataset))
    print(dataset[0])
