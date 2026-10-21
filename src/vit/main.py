from trainer import ViTClassifier
from dataset import Flowers102FewShot
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utils import accuracy

train_ds = Flowers102FewShot(root="data", split="train")
val_ds = Flowers102FewShot(root="data", split="val")
test_ds = Flowers102FewShot(root="data", split="test")

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)
test_loader = DataLoader(test_ds, batch_size=16)

tt = "zs"

model = ViTClassifier(
    model_name="google/vit-base-patch16-224",
    lr=1e-4,
    weight_decay=0.05,
    training_type=tt,  # Choose from ["peft", "full", "lp", "zs"]
    num_classes=102,
    metric_fns={"acc": accuracy}
)

csv_logger = pl.loggers.CSVLogger("src/vit/logs", name=f"vit_classifier_{tt}")
callbacks = [
    pl.callbacks.ModelCheckpoint(monitor="val/loss", save_top_k=1, mode="min",
                                 filename="best-{epoch:02d}-{val/loss:.4f}", auto_insert_metric_name=False),
    pl.callbacks.EarlyStopping("val/loss", patience=10),
    pl.callbacks.LearningRateMonitor("step")
]

Trainer = pl.Trainer(
    max_epochs=30,
    accelerator="auto",
    devices="auto",
    logger=csv_logger,
    callbacks=callbacks,
)

# Trainer.fit(model, train_loader, val_loader)
# Trainer.test(model, test_loader, ckpt_path="best")

Trainer.test(model, test_loader)