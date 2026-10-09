from __future__ import annotations

import math

import lightning as L
import torch
import timm
from torch import nn
from torch.nn import functional as F
from transformers import ViTImageProcessor, ViTForImageClassification

METHODS = ("zero_shot", "linear", "full", "bitfit")


class ViTClassifier(L.LightningModule):
    def __init__(
        self, num_classes, method, learning_rate, weight_decay, max_epochs):
        super().__init__()

        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")

        self.save_hyperparameters()
        output_classes = 1000 if method == "zero_shot" else num_classes

        # ===== Model =======

        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            ignore_mismatched_sizes = True,
            num_labels = output_classes,
            output_attentions = True,
            output_hidden_states = True,
            attn_implementation = "eager"
        )
        
        # Losses
        self.criterion = nn.CrossEntropyLoss()

        
        self.num_classes = num_classes
        self.output_classes = output_classes
        self.test_predictions = []
        self.test_targets = []
        

    def _configure_trainable_parameters(self, method):

        if method in ("zero_shot", "full"):
            for parameter in self.parameters():
                parameter.requires_grad = method == "full"
            return

        self.requires_grad_(False)

        if method == "linear":
            self.model.get_classifier().requires_grad_(True)

        elif method == "bitfit":
            for name, parameter in self.named_parameters():
                parameter.requires_grad = name.endswith(".bias") and "classifier" in name
            self.model.get_classifier().requires_grad_(True)


    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs.logits, outputs.hidden_states[-1][:, 0, :]

    def _shared_step(self, batch, stage, batch_idx):
        inputs, targets = batch
        logits, _ = self(inputs)

        # =====Loss and accuracy =========
        loss = self.criterion(logits, targets)
        accuracy = (logits.argmax(dim=1) == targets).float().mean()

        # ====== Logging ==========
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}/accuracy", accuracy, prog_bar=True, on_epoch=True)

        if stage == "train" and batch_idx is not None and (batch_idx + 1) % 10 == 0:
            self.log("train/loss_step", loss, on_step=True, prog_bar=True)
            self.log("train/accuracy_step", accuracy, on_step=True, prog_bar=True)

        return loss


    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train", batch_idx)


    def validation_step( self, batch, batch_idx):
        self._shared_step(batch, "val", batch_idx)


    def on_test_start(self):
        self.test_predictions.clear()
        self.test_targets.clear()


    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)

        #===== Store predictions, embeddings, and targets =====
        predictions = logits.argmax(dim=1)

        self.test_predictions.append(predictions.detach().cpu())
        self.test_targets.append(targets.detach().cpu())

        #===== Logging =====
        if self.output_classes == self.num_classes:
            self.log("test/accuracy", (predictions == targets).float().mean(), on_epoch=True)


    @property
    def trainable_parameters(self):
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            (parameter for parameter in self.parameters() if parameter.requires_grad),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}