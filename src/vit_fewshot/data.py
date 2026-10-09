from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

DATASET_CLASSES = {"cifar100": 100, "flowers102": 102}


def _labels(dataset: Dataset) -> list[int]:
    labels = getattr(dataset, "targets", None)
    if labels is None:
        labels = getattr(dataset, "_labels", None)
    if labels is None:
        raise ValueError(f"Cannot find labels on {type(dataset).__name__}")
    return [int(label) for label in labels]


def stratified_indices(
    labels: list[int], samples_per_class: int, seed: int, excluded: set[int] | None = None
) -> list[int]:
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be positive")

    grouped: dict[int, list[int]] = defaultdict(list)
    excluded = excluded or set()
    for index, label in enumerate(labels):
        if index not in excluded:
            grouped[label].append(index)

    rng = random.Random(seed)
    selected: list[int] = []
    for label, indices in sorted(grouped.items()):
        if len(indices) < samples_per_class:
            raise ValueError(
                f"Class {label} has {len(indices)} samples, requested {samples_per_class}"
            )
        selected.extend(rng.sample(indices, samples_per_class))
    return selected


class FewShotDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: str,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        shots_per_class: int,
        seed: int,
    ) -> None:
        super().__init__()
        if dataset not in DATASET_CLASSES:
            raise ValueError(f"Unknown dataset: {dataset}")
        self.dataset_name = dataset
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shots_per_class = shots_per_class
        self.seed = seed

        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.eval_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def prepare_data(self) -> None:
        if self.dataset_name == "cifar100":
            datasets.CIFAR100(self.data_dir, train=True, download=True)
            datasets.CIFAR100(self.data_dir, train=False, download=True)
            
        elif self.dataset_name == "flowers102":
            datasets.Flowers102(self.data_dir, split="train", download=True)
            datasets.Flowers102(self.data_dir, split="val", download=True)
            datasets.Flowers102(self.data_dir, split="test", download=True)

            for split in ("train", "val", "test"):
                datasets.Flowers102(self.data_dir, split=split, download=True)

    def setup(self, stage: str | None = None) -> None:
        if self.dataset_name == "cifar100":
            self.train_dataset = datasets.CIFAR100(
                self.data_dir, train=True, transform=self.train_transform
            )
            self.val_dataset = datasets.CIFAR100(
                self.data_dir, train=True, transform=self.eval_transform
            )
            self.test_dataset = datasets.CIFAR100(
                self.data_dir, train=False, transform=self.eval_transform
            )
        elif self.dataset_name == "flowers102":
            self.train_dataset = datasets.Flowers102(
                self.data_dir, split="train", transform=self.train_transform
            )
            self.val_dataset = datasets.Flowers102(
                self.data_dir, split="val", transform=self.eval_transform
            )
            self.test_dataset = datasets.Flowers102(
                self.data_dir, split="test", transform=self.eval_transform
            )

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)
