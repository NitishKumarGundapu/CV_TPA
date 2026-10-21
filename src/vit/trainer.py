import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Callable, Dict, Optional, Sequence
from transformers import ViTForImageClassification
from peft import get_peft_model, LoraConfig


class ViTClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        metric_fns: Optional[Dict[str, Callable]] = None,
        training_type: str = "lp", #["peft", "full", "lp", "zs"]
        num_classes: int = 10,

    ):
        super().__init__()

        if training_type == "zs":
            self.model = ViTForImageClassification.from_pretrained(
                model_name,
                output_attentions=False,
            )
        else:
            self.model = ViTForImageClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                output_attentions=False,
                ignore_mismatched_sizes=True
            )

        self.loss_fn = nn.CrossEntropyLoss()
        self.metric_fns = metric_fns or {}
        self.lr = lr
        self.weight_decay = weight_decay

        if training_type == "peft":
            peft_config = LoraConfig(
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],
                modules_to_save=["classifier"],
            )
            self.model = get_peft_model(self.model, peft_config)

        if training_type in ["full", "lp"]:
            self.freeze_parameters(training_type)

        self.save_hyperparameters(ignore=["model", "loss_fn", "metric_fns"])


    def freeze_parameters(self, training_type: str):
        if training_type == "full":
            for param in self.model.parameters():
                param.requires_grad = True
        
        elif training_type == "lp":
            for name, param in self.model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False


    def forward(self, x):
        return self.model(pixel_values=x).logits


    def _shared_step(self, batch, stage: str):
        pixel_values, targets = batch["pixel_values"], batch["labels"]
        preds = self.forward(pixel_values)
        loss = self.loss_fn(preds, targets)
        self.log(
            f"{stage}/loss", loss, prog_bar=True,
            on_step=(stage == "train"), on_epoch=True, sync_dist=True,
        )

        if stage != "train":
            for name, fn in self.metric_fns.items():
                self.log(
                    f"{stage}/{name}", fn(preds.detach().float(), targets),
                    prog_bar=True, on_step=False, on_epoch=True, sync_dist=True,
                )
        return loss


    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")


    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")


    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pixel_values = batch["pixel_values"]
        preds = self.forward(pixel_values)
        # add the softmax here
        return preds


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            (parameter for parameter in self.parameters() if parameter.requires_grad),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        max_epochs = self.trainer.max_epochs
        t_max = max_epochs if max_epochs is not None and max_epochs > 0 else 1
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(t_max),
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

