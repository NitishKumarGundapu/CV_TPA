from __future__ import annotations

from collections import defaultdict
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import Flowers102
from transformers import AutoImageProcessor

from .config import Config, resolve_path


def processor_size(image_processor) -> int:
    size = getattr(image_processor, "size", 224)
    if isinstance(size, dict):
        return int(size.get("height") or size.get("shortest_edge") or 224)
    if getattr(size, "height", None):
        return int(size.height)
    if isinstance(size, (tuple, list)):
        return int(size[0])
    return int(size)


def processor_stats(image_processor) -> tuple[tuple[float, ...], tuple[float, ...]]:
    mean = getattr(image_processor, "image_mean", None) or (0.5, 0.5, 0.5)
    std = getattr(image_processor, "image_std", None) or (0.5, 0.5, 0.5)
    return tuple(float(value) for value in mean), tuple(float(value) for value in std)


class Flowers102FewShot(Dataset):
    """Oxford Flowers102 formatted for ``ViTClassifier``.

    The official training split contains 10 images per class. ``shots_per_class``
    subsamples it deterministically (the same seed always yields the same indices).
    Validation and test always use their complete official splits.
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
        augment: bool = False,
    ):
        if split not in self._SPLITS:
            raise ValueError(f"split must be one of {tuple(self._SPLITS)}, got {split!r}")
        if shots_per_class is not None and shots_per_class < 1:
            raise ValueError("shots_per_class must be a positive integer or None")

        self.dataset = Flowers102(
            root=str(root),
            split=self._SPLITS[split],
            download=download,
        )
        self.image_processor = image_processor or AutoImageProcessor.from_pretrained(model_name)
        self.indices = list(range(len(self.dataset)))

        if shots_per_class is not None and self._SPLITS[split] == "train":
            by_class: dict[int, list[int]] = defaultdict(list)
            for index, label in enumerate(self._all_labels()):
                by_class[int(label)].append(index)
            for label, label_indices in sorted(by_class.items()):
                if len(label_indices) < shots_per_class:
                    raise ValueError(
                        f"class {label} only has {len(label_indices)} training images, "
                        f"requested {shots_per_class}"
                    )
            generator = torch.Generator().manual_seed(seed)
            selected: list[int] = []
            for label in sorted(by_class):
                label_indices = by_class[label]
                order = torch.randperm(len(label_indices), generator=generator).tolist()
                selected.extend(label_indices[position] for position in order[:shots_per_class])
            self.indices = sorted(selected)

        self.transform = self._build_augmentation() if augment else None

    def _all_labels(self) -> list[int]:
        labels = getattr(self.dataset, "_labels", None)
        if labels is None or len(labels) != len(self.dataset):
            labels = [int(self.dataset[index][1]) for index in range(len(self.dataset))]
        return [int(label) for label in labels]

    def _build_augmentation(self) -> transforms.Compose:
        size = processor_size(self.image_processor)
        mean, std = processor_stats(self.image_processor)
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size, scale=(0.65, 1.0), ratio=(0.75, 1.333)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    @property
    def labels(self) -> list[int]:
        all_labels = self._all_labels()
        return [all_labels[index] for index in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image, label = self.dataset[self.indices[index]]
        image = image.convert("RGB")
        if self.transform is not None:
            pixel_values = self.transform(image)
        else:
            pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(int(label), dtype=torch.long),
        }


def build_datasets(cfg: Config):
    root = resolve_path(cfg.data_root)
    processor = AutoImageProcessor.from_pretrained(cfg.model_name)
    common = dict(
        root=str(root),
        seed=cfg.seed,
        model_name=cfg.model_name,
        image_processor=processor,
        download=True,
    )
    train = Flowers102FewShot(
        split="train", shots_per_class=cfg.shots_per_class, augment=True, **common
    )
    train_eval = Flowers102FewShot(
        split="train", shots_per_class=cfg.shots_per_class, augment=False, **common
    )
    val = Flowers102FewShot(split="val", shots_per_class=None, **common)
    test = Flowers102FewShot(split="test", shots_per_class=None, **common)
    return processor, train, train_eval, val, test


def _loader(dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def build_dataloaders(cfg: Config, train, train_eval, val, test) -> dict[str, DataLoader]:
    return {
        "train": _loader(train, cfg.batch_size, True, cfg.num_workers, cfg.seed),
        "train_eval": _loader(train_eval, cfg.batch_size, False, cfg.num_workers, cfg.seed),
        "val": _loader(val, cfg.batch_size, False, cfg.num_workers, cfg.seed),
        "test": _loader(test, cfg.batch_size, False, cfg.num_workers, cfg.seed),
    }
