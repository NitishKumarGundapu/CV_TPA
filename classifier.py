# --- image classification: one function + one wrapper ---
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Make `import src...` work no matter where the notebook is launched from.
_REPO_ROOT = Path.cwd()
while not (_REPO_ROOT / "src" / "vit").is_dir() and _REPO_ROOT != _REPO_ROOT.parent:
    _REPO_ROOT = _REPO_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.dino_v3.dataset import build_transforms as build_dino_transforms
from src.dino_v3.evaluate import load_checkpoint as load_dino_checkpoint
from src.dino_v3.model import DINOv3Classifier
from src.vit.evaluate import load_checkpoint as load_vit_checkpoint
from src.vit.evaluate import predict_prototypes
from src.vit.trainer import ViTClassifier

METHODS = ("full", "lp", "lora", "bitfit", "ncm")
_METHOD_ALIASES = {"peft": "lora", "zs": "ncm", "zero-shot": "ncm", "linear": "lp"}
_DATASET_ALIASES = {
    "flowers": "flowers102",
    "pets": "oxfordiiitpet",
    "oxford-pets": "oxfordiiitpet",
    "oxford_iiit_pet": "oxfordiiitpet",
}
_FLOWERS102_NAMES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea",
    "english marigold", "tiger lily", "moon orchid", "bird of paradise", "monkshood",
    "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle",
    "yellow iris", "globe-flower", "purple coneflower", "peruvian lily",
    "balloon flower", "giant white arum lily", "fire lily", "pincushion flower",
    "fritillary", "red ginger", "grape hyacinth", "corn poppy",
    "prince of wales feathers", "stemless gentian", "artichoke", "sweet william",
    "carnation", "garden phlox", "love in the mist", "mexican aster",
    "alpine sea holly", "ruby-lipped cattleya", "cape flower", "great masterwort",
    "siam tulip", "lenten rose", "barbeton daisy", "daffodil", "sword lily",
    "poinsettia", "bolero deep blue", "wallflower", "marigold", "buttercup",
    "oxeye daisy", "common dandelion", "petunia", "wild pansy", "primula",
    "sunflower", "pelargonium", "bishop of llandaff", "gaura", "geranium",
    "orange dahlia", "pink-yellow dahlia?", "cautleya spicata", "japanese anemone",
    "black-eyed susan", "silverbush", "californian poppy", "osteospermum",
    "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania",
    "azalea", "water lily", "rose", "thorn apple", "morning glory",
    "passion flower", "lotus", "toad lily", "anthurium", "frangipani", "clematis",
    "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia", "cyclamen",
    "watercress", "canna lily", "hippeastrum", "bee balm", "ball moss", "foxglove",
    "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia",
    "blanket flower", "trumpet creeper", "blackberry lily",
]


def _normalize_method(method: str) -> str:
    method = method.strip().lower()
    method = _METHOD_ALIASES.get(method, method)
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}. Choose from {METHODS}")
    return method


def _normalize_dataset(dataset: str) -> str:
    dataset = dataset.strip().lower()
    dataset = _DATASET_ALIASES.get(dataset, dataset)
    if dataset not in ("flowers102", "oxfordiiitpet"):
        raise ValueError(f"Unknown dataset {dataset!r}. Use flowers102 or oxfordiiitpet")
    return dataset


def get_class_names(dataset: str, num_classes: int | None = None) -> list[str]:
    """Human-readable class names, ordered the same way as the training labels."""
    dataset = _normalize_dataset(dataset)
    if dataset == "flowers102":
        return list(_FLOWERS102_NAMES)
    folder = _REPO_ROOT / "data" / "oxford_pet" / "test"
    if folder.is_dir():
        from torchvision.datasets import ImageFolder

        return [name.replace("_", " ") for name in ImageFolder(str(folder)).classes]
    if num_classes is None:
        num_classes = 37
    return [f"class_{i}" for i in range(num_classes)]


def _infer_family(name: str) -> str | None:
    name = name.lower()
    if "dinov3" in name or "dino_v3" in name or name == "dino":
        return "dino"
    if name == "vit" or name.startswith(("google/", "vit-")):
        return "vit"
    return None


def _detect_family(state_dict) -> str:
    if any(key.startswith("backbone.") for key in state_dict):
        return "dino"
    if any(key.startswith("model.") for key in state_dict):
        return "vit"
    raise ValueError(
        "Unrecognised checkpoint: expected `backbone.*` (DINOv3) or `model.*` (ViT) weights"
    )


def _artifact_in_dir(run_dir: Path, method: str) -> Path:
    if method == "ncm":
        files = sorted(run_dir.rglob("prototypes.npz"))
    else:
        files = sorted(run_dir.rglob("*.ckpt"))
        files = [f for f in files if "best" in f.name] or files
    if not files:
        expected = "prototypes.npz" if method == "ncm" else "*.ckpt"
        raise FileNotFoundError(f"No {expected} under {run_dir}")
    return max(files, key=lambda path: path.stat().st_mtime)


def resolve_artifact(
    model_name: str | Path,
    dataset: str,
    method: str,
    seed: int = 42,
    results_dir: str | Path | None = None,
) -> Path:
    """Find the `.ckpt` / `prototypes.npz` for a run.

    `model_name` may be a file, a run directory, a family label ("vit"/"dinov3")
    or a base model name; non-file names are looked up under ``results/``.
    """
    path = Path(str(model_name)).expanduser()
    if path.exists():
        return _artifact_in_dir(path, method) if path.is_dir() else path

    family = _infer_family(str(model_name))
    if results_dir is not None:
        base = Path(results_dir).expanduser()
        if (base / "vit").is_dir() or (base / "dino_v3").is_dir():
            roots = [base / "vit", base / "dino_v3"]
        else:
            roots = [base]
    else:
        roots = [_REPO_ROOT / "results" / "vit", _REPO_ROOT / "results" / "dino_v3"]
    if family == "vit":
        roots = [root for root in roots if root.name == "vit"]
    elif family == "dino":
        roots = [root for root in roots if root.name == "dino_v3"]

    candidates = []
    for root in roots:
        run_dir = root / dataset / method / f"seed_{seed}"
        if run_dir.is_dir():
            try:
                candidates.append(_artifact_in_dir(run_dir, method))
            except FileNotFoundError:
                pass
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)

    raise FileNotFoundError(
        f"No {method!r} artifact for dataset={dataset!r}, seed={seed} "
        f"(looked for model_name={str(model_name)!r}). Pass a .ckpt / "
        "prototypes.npz path or a run directory."
    )


def _load_artifact(source: Path) -> tuple[str, str, int, torch.nn.Module, np.ndarray | None]:
    if source.suffix == ".npz":
        payload = np.load(source, allow_pickle=False)
        prototypes = payload["prototypes"].astype(np.float32)
        family = "dino" if "model_name" in payload.files else "vit"
        model_name = (
            str(payload["model_name"].item())
            if "model_name" in payload.files
            else "google/vit-base-patch16-224"
        )
        num_classes = (
            int(payload["num_classes"].item())
            if "num_classes" in payload.files
            else int(prototypes.shape[0])
        )
        model_cls = DINOv3Classifier if family == "dino" else ViTClassifier
        model = model_cls(
            model_name=model_name,
            num_classes=num_classes,
            method="ncm",
            metric_fns=None,
        )
        return family, model_name, num_classes, model, prototypes

    payload = torch.load(source, map_location="cpu", weights_only=False)
    family = _detect_family(payload.get("state_dict", {}))
    loader = load_dino_checkpoint if family == "dino" else load_vit_checkpoint
    model = loader(source)
    model_name = getattr(model, "model_name", None) or model.hparams.get("model_name")
    return family, model_name, int(model.num_classes), model, None


def _build_preprocess(family: str, model_name: str):
    if family == "dino":
        transform = build_dino_transforms(model_name, augment=False)

        def preprocess(image):
            return transform(image.convert("RGB")).unsqueeze(0)

        return preprocess

    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(model_name)

    def preprocess(image):
        return processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"]

    return preprocess


@torch.inference_mode()
def predict_top_k(
    model,
    preprocess,
    image_path,
    class_names: list[str],
    *,
    method: str = "lp",
    prototypes: np.ndarray | None = None,
    top_k: int = 5,
    device=None,
) -> list[dict]:
    """Core: run one image through a loaded model and return the top-k classes."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()

    image = (
        image_path
        if isinstance(image_path, Image.Image)
        else Image.open(image_path).convert("RGB")
    )
    pixel_values = preprocess(image).to(device)

    logits, features = model.encode_with_logits(pixel_values)
    logits = logits.float().cpu().numpy()
    if method == "ncm" or prototypes is not None:
        if prototypes is None:
            raise ValueError("method='ncm' requires prototypes")
        logits = predict_prototypes(features.float().cpu().numpy(), prototypes)

    probs = torch.from_numpy(logits).softmax(dim=-1)[0].numpy()
    order = np.argsort(probs)[::-1][: min(top_k, len(probs))]
    return [
        {
            "rank": rank + 1,
            "class_id": int(class_id),
            "class_name": (
                class_names[class_id]
                if class_id < len(class_names)
                else f"class_{class_id}"
            ),
            "confidence": float(probs[class_id]),
        }
        for rank, class_id in enumerate(order)
    ]


def classify_image(
    model_name: str | Path,
    dataset: str,
    image_path,
    method: str,
    *,
    top_k: int = 5,
    seed: int = 42,
    results_dir: str | Path | None = None,
    class_names: list[str] | None = None,
    device=None,
) -> list[dict]:
    """Wrapper: locate/load the right model for `method` and return top-k classes.

    Args:
        model_name:  `.ckpt`, `prototypes.npz`, run directory, family label
                     ("vit"/"dinov3") or base model name.
        dataset:     "flowers102" or "oxfordiiitpet" (class-name lookup).
        image_path:  path to an image (or a PIL.Image).
        method:      "full" | "lp" | "lora" | "bitfit" | "ncm".
        top_k:       number of classes to return.
    """
    method = _normalize_method(method)
    dataset = _normalize_dataset(dataset)

    source = resolve_artifact(model_name, dataset, method, seed, results_dir)
    family, backbone_name, num_classes, model, prototypes = _load_artifact(source)

    trained_method = getattr(model, "method", method)
    if method != "ncm" and trained_method != method:
        raise ValueError(
            f"{source} was trained with method={trained_method!r}, not {method!r}"
        )
    if method == "ncm" and prototypes is None:
        raise ValueError(
            f"method='ncm' needs a prototypes.npz, got checkpoint {source}"
        )

    names = class_names or get_class_names(dataset, num_classes)
    preprocess = _build_preprocess(family, backbone_name)
    return predict_top_k(
        model,
        preprocess,
        image_path,
        names,
        method=method,
        prototypes=prototypes,
        top_k=top_k,
        device=device,
    )
