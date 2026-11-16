from __future__ import annotations

from collections import defaultdict
from typing import Optional

import timm
import torch
from timm.data import create_transform, resolve_data_config
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Flowers102

from .config import Config, resolve_path


def resolve_model_data_config(model_name: str) -> dict:
    pretrained_cfg = timm.models.get_pretrained_cfg(model_name)
    if pretrained_cfg is None:
        raise ValueError(
            f"Unknown timm model {model_name!r}. Pass a registered DINOv3 model such as "
            "'vit_base_patch16_dinov3.lvd1689m'."
        )
    return resolve_data_config(pretrained_cfg=pretrained_cfg.to_dict())


def build_transforms(model_name: str, augment: bool):
    data_config = resolve_model_data_config(model_name)
    if augment:
        return create_transform(
            **data_config,
            is_training=True,
            scale=(0.65, 1.0),
            ratio=(0.75, 1.3333),
            hflip=0.5,
            color_jitter=0.0,
            auto_augment=None,
        )
    return create_transform(**data_config, is_training=False)


class Flowers102FewShot(Dataset):
    """Oxford Flowers102 formatted for the DINOv3 classifier.

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
        transform=None,
        download: bool = True,
    ):
        if split not in self._SPLITS:
            raise ValueError(f"split must be one of {tuple(self._SPLITS)}, got {split!r}")
        if shots_per_class is not None and shots_per_class < 1:
            raise ValueError("shots_per_class must be a positive integer or None")
        if transform is None:
            raise ValueError("transform is required")

        self.dataset = Flowers102(
            root=str(root),
            split=self._SPLITS[split],
            download=download,
        )
        self.transform = transform
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

    def _all_labels(self) -> list[int]:
        labels = getattr(self.dataset, "_labels", None)
        if labels is None or len(labels) != len(self.dataset):
            labels = [int(self.dataset[index][1]) for index in range(len(self.dataset))]
        return [int(label) for label in labels]

    @property
    def labels(self) -> list[int]:
        all_labels = self._all_labels()
        return [all_labels[index] for index in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image, label = self.dataset[self.indices[index]]
        image = image.convert("RGB")
        return {
            "pixel_values": self.transform(image),
            "labels": torch.tensor(int(label), dtype=torch.long),
        }


def build_datasets(cfg: Config):
    root = resolve_path(cfg.data_root)
    train_transform = build_transforms(cfg.model_name, augment=True)
    eval_transform = build_transforms(cfg.model_name, augment=False)
    common = dict(root=str(root), seed=cfg.seed, download=True)

    train = Flowers102FewShot(
        split="train",
        shots_per_class=cfg.shots_per_class,
        transform=train_transform,
        **common,
    )
    train_eval = Flowers102FewShot(
        split="train",
        shots_per_class=cfg.shots_per_class,
        transform=eval_transform,
        **common,
    )
    val = Flowers102FewShot(
        split="val", shots_per_class=None, transform=eval_transform, **common
    )
    test = Flowers102FewShot(
        split="test", shots_per_class=None, transform=eval_transform, **common
    )
    return train, train_eval, val, test


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
