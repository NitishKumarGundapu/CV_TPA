from __future__ import annotations

from typing import Callable, Optional

import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model

from .config import normalize_method
from .metrics import accuracy, top_k_accuracy


class DINOv3Classifier(pl.LightningModule):
    """DINOv3 backbone with a linear classification head.

    Methods:
        full    - update every parameter
        lp      - linear probe: update the classification head only
        lora    - LoRA adapters on attention qkv/proj plus the classification head
        bitfit  - update bias terms and the classification head
        ncm     - training-free nearest-class-mean over frozen CLS features
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_dinov3.lvd1689m",
        num_classes: int = 102,
        method: str = "lp",
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        metric_fns: Optional[dict[str, Callable]] = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.method = normalize_method(method)
        self.model_name = model_name
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.metric_fns = metric_fns or {"acc": accuracy, "top5": top_k_accuracy}

        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool="",
        )
        self.feature_dim = int(self.backbone.num_features)
        self.head = nn.Linear(self.feature_dim, num_classes)

        if self.method == "lora":
            peft_config = LoraConfig(
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["attn.qkv", "attn.proj"],
            )
            self.backbone = get_peft_model(self.backbone, peft_config)

        self.configure_trainable_parameters()
        self.loss_fn = nn.CrossEntropyLoss()
        self.save_hyperparameters(ignore=["metric_fns"])

    def configure_trainable_parameters(self) -> None:
        if self.method == "lora":
            for parameter in self.head.parameters():
                parameter.requires_grad = True
            return

        for parameter in self.parameters():
            parameter.requires_grad = False

        if self.method == "full":
            for parameter in self.backbone.parameters():
                parameter.requires_grad = True
            for parameter in self.head.parameters():
                parameter.requires_grad = True
        elif self.method == "lp":
            for parameter in self.head.parameters():
                parameter.requires_grad = True
        elif self.method == "bitfit":
            for name, parameter in self.backbone.named_parameters():
                if name.endswith("bias"):
                    parameter.requires_grad = True
            for parameter in self.head.parameters():
                parameter.requires_grad = True
        elif self.method != "ncm":
            raise ValueError(f"Unknown method {self.method!r}")

    def parameter_counts(self) -> dict[str, float]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "trainable_fraction": float(trainable / total) if total else 0.0,
        }

    def get_backbone(self):
        if isinstance(self.backbone, PeftModel):
            return self.backbone.get_base_model()
        return self.backbone

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.get_backbone().forward_features(pixel_values)
        return features[:, 0]

    def encode_with_logits(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(pixel_values)
        return self.head(features), features

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(pixel_values))

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str):
        pixel_values = batch["pixel_values"]
        targets = batch["labels"]
        logits = self(pixel_values)
        loss = self.loss_fn(logits, targets)
        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
            sync_dist=True,
        )

        if stage != "train":
            for name, fn in self.metric_fns.items():
                self.log(
                    f"{stage}/{name}",
                    fn(logits.detach().float(), targets),
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(batch["pixel_values"])
        return {
            "logits": logits,
            "probs": logits.softmax(dim=-1),
            "labels": batch["labels"],
        }

    def configure_optimizers(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        if not parameters:
            return None

        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        max_epochs = getattr(self.trainer, "max_epochs", None)
        t_max = max_epochs if max_epochs is not None and max_epochs > 0 else 1
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(t_max),
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
