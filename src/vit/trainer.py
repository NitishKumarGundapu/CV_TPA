from __future__ import annotations

from typing import Callable, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import ViTForImageClassification

from .config import normalize_method
from .metrics import accuracy, top_k_accuracy


class ViTClassifier(pl.LightningModule):
    """ViT image classifier with switchable adaptation strategies.

    Methods:
        full    - update every parameter
        lp      - linear probe: update the classification head only
        lora    - LoRA adapters on q_proj/v_proj plus the classification head
        bitfit  - update bias terms and the classification head
        ncm     - training-free nearest-class-mean over frozen CLS features
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
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
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.metric_fns = metric_fns or {"acc": accuracy, "top5": top_k_accuracy}

        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

        if self.method == "lora":
            peft_config = LoraConfig(
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "v_proj"],
                modules_to_save=["classifier"],
            )
            self.model = get_peft_model(self.model, peft_config)

        if self.method in ("full", "lp", "bitfit", "ncm"):
            self.freeze_parameters(self.method)

        self.loss_fn = nn.CrossEntropyLoss()
        self.save_hyperparameters(ignore=["model", "loss_fn", "metric_fns"])

    def freeze_parameters(self, method: str) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        if method == "full":
            for parameter in self.model.parameters():
                parameter.requires_grad = True
        elif method == "lp":
            for name, parameter in self.model.named_parameters():
                if "classifier" in name:
                    parameter.requires_grad = True
        elif method == "bitfit":
            for name, parameter in self.model.named_parameters():
                if "classifier" in name or name.endswith("bias"):
                    parameter.requires_grad = True
        elif method != "ncm":
            raise ValueError(f"Unknown method {method!r}")

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

    def get_backbone(self) -> ViTForImageClassification:
        model = self.model
        if isinstance(model, PeftModel):
            return model.get_base_model()
        return model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).logits

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.get_backbone().vit(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 0]

    def encode_with_logits(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        backbone = self.get_backbone()
        features = backbone.vit(pixel_values=pixel_values).last_hidden_state[:, 0]
        return backbone.classifier(features), features

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
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

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
