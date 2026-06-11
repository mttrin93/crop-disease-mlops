"""
EfficientNet-B0 training script for EC2 / Airflow.

Mirrors the logic of notebooks/train_colab.ipynb but as a CLI script
that can be called by an Airflow BashOperator.

Usage:
    python src/model/train.py \
        --data-dir data/processed \
        --epochs 15 \
        --batch-size 64 \
        --mlflow-uri http://your-ec2:5000 \
        --experiment-name crop-disease-detection \
        --model-name crop-disease-efficientnet-b0

Prints the RUN_ID to stdout on completion so Airflow can capture it
via XCom and keep SSM in sync automatically.
"""

import io
import os
import json
import time
import logging
import argparse
from pathlib import Path

import onnx  # pylint: disable=import-error
import boto3
import numpy as np
import torch
import mlflow
import pandas as pd
import mlflow.onnx
import mlflow.pytorch
from torch import nn, optim
from torchvision import transforms
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.datasets import ImageFolder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------- data ----------


def build_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    train_ds = ImageFolder(root=str(data_dir / "train"), transform=train_transforms)
    val_ds = ImageFolder(root=str(data_dir / "val"), transform=eval_transforms)
    test_ds = ImageFolder(root=str(data_dir / "test"), transform=eval_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        "Train: %s | Val: %s | Test: %s", len(train_ds), len(val_ds), len(test_ds)
    )
    return train_loader, val_loader, test_loader


# ---------- model ----------


def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model.to(DEVICE)


# ---------- training ----------


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return total_loss / total, correct / total, f1


# ---------- ONNX export ----------


def export_onnx(model: nn.Module, output_path: str) -> None:
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=18,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    logger.info("ONNX model exported → %s", output_path)


# ---------- reference data for monitoring ----------


def generate_reference_data(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    model: nn.Module,
    test_loader: DataLoader,
    class_names: list[str],
    output_dir: Path,
    model_bucket: str | None = None,
    s3_key: str = "monitoring/reference_predictions.parquet",
) -> None:
    """
    Run the trained model on the full test set and save the prediction
    distribution to S3 as a Parquet file.

    This becomes the REFERENCE DATA for Evidently drift monitoring.
    Every day, recent API predictions are compared against this baseline
    to detect distribution shift.

    Columns saved:
        predicted_class  — top-1 predicted class name
        confidence       — top-1 softmax probability
        entropy          — prediction entropy (uncertainty measure)
    """
    model.eval()
    records = []

    with torch.no_grad():
        for images, _ in test_loader:
            probs_batch = torch.softmax(model(images.to(DEVICE)), dim=1).cpu().numpy()
            for probs in probs_batch:
                top_idx = int(probs.argmax())
                entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
                records.append(
                    {
                        "predicted_class": class_names[top_idx],
                        "confidence": float(probs[top_idx]),
                        "entropy": entropy,
                    }
                )

    reference_df = pd.DataFrame(records)
    logger.info(
        "Reference data: %d predictions, %d unique classes",
        len(reference_df),
        reference_df["predicted_class"].nunique(),
    )

    local_path = output_dir / "reference_predictions.parquet"
    reference_df.to_parquet(local_path, index=False)
    mlflow.log_artifact(str(local_path), artifact_path="monitoring")

    if model_bucket:
        buf = io.BytesIO()
        reference_df.to_parquet(buf, index=False)
        buf.seek(0)
        boto3.client("s3").put_object(
            Bucket=model_bucket,
            Key=s3_key,
            Body=buf.getvalue(),
        )
        logger.info("Reference data uploaded to s3://%s/%s", model_bucket, s3_key)


# ---------- training loop helpers ----------


class _EpochContext:  # pylint: disable=too-few-public-methods
    """Groups per-epoch training state to avoid long argument lists."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler


class _ArtifactContext:  # pylint: disable=too-few-public-methods,too-many-instance-attributes
    """Groups artifact paths and identifiers for final logging."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        test_loader: DataLoader,
        criterion: nn.Module,
        data_dir: Path,
        output_dir: Path,
        best_ckpt: str,
        model_name: str,
        class_names: list[str] | None = None,
        model_bucket: str | None = None,
    ) -> None:
        self.test_loader = test_loader
        self.criterion = criterion
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.best_ckpt = best_ckpt
        self.model_name = model_name
        self.class_names = class_names or []
        self.model_bucket = model_bucket


def _make_optimizer(
    model: nn.Module, lr: float, weight_decay: float
) -> optim.Optimizer:
    return optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )


def _make_scheduler(optimizer: optim.Optimizer) -> optim.lr_scheduler.ReduceLROnPlateau:
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )


def _unfreeze_backbone(
    model: nn.Module, lr: float, weight_decay: float
) -> tuple[optim.Optimizer, optim.lr_scheduler.ReduceLROnPlateau]:
    logger.info("Unfreezing backbone...")
    for param in model.parameters():
        param.requires_grad = True
    new_optimizer = optim.AdamW(
        model.parameters(), lr=lr / 10, weight_decay=weight_decay
    )
    new_scheduler = _make_scheduler(new_optimizer)
    return new_optimizer, new_scheduler


def _run_epoch(
    epoch: int, num_epochs: int, ctx: _EpochContext
) -> tuple[float, float, float, float, float]:
    """Run one train + val epoch. Returns train_loss, train_acc, val_loss, val_acc, val_f1."""
    t0 = time.time()
    train_loss, train_acc = train_one_epoch(
        ctx.model, ctx.train_loader, ctx.criterion, ctx.optimizer
    )
    val_loss, val_acc, val_f1 = evaluate(ctx.model, ctx.val_loader, ctx.criterion)
    ctx.scheduler.step(val_loss)

    logger.info(
        "Epoch %02d/%d | train loss %.4f acc %.4f | val loss %.4f acc %.4f f1 %.4f | %.1fs",
        epoch,
        num_epochs,
        train_loss,
        train_acc,
        val_loss,
        val_acc,
        val_f1,
        time.time() - t0,
    )
    return train_loss, train_acc, val_loss, val_acc, val_f1


def _log_and_register(model: nn.Module, ctx: _ArtifactContext) -> None:
    """Evaluate on test set, export ONNX, log and register in MLflow."""
    model.load_state_dict(torch.load(ctx.best_ckpt, map_location=DEVICE))
    test_loss, test_acc, test_f1 = evaluate(model, ctx.test_loader, ctx.criterion)
    logger.info("Test accuracy: %.4f | Test F1: %.4f", test_acc, test_f1)
    mlflow.log_metrics(
        {"test_loss": test_loss, "test_acc": test_acc, "test_f1": test_f1}
    )

    onnx_path = str(ctx.output_dir / "model.onnx")
    export_onnx(model, onnx_path)
    onnx_model = onnx.load(onnx_path)
    mlflow.onnx.log_model(onnx_model=onnx_model, artifact_path="onnx")
    mlflow.log_artifact(str(ctx.data_dir / "metadata.json"), artifact_path="onnx")

    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        registered_model_name=ctx.model_name,
    )

    # generate and upload reference data for Evidently monitoring
    if ctx.class_names:
        generate_reference_data(
            model=model,
            test_loader=ctx.test_loader,
            class_names=ctx.class_names,
            output_dir=ctx.output_dir,
            model_bucket=ctx.model_bucket,
        )


def _run_training_loop(
    model: nn.Module,
    ctx: _EpochContext,
    artifact_ctx: _ArtifactContext,
    num_epochs: int,
    early_stop_patience: int,
) -> None:
    """Run the full training loop and log final artifacts."""
    best_val_loss = float("inf")
    epochs_no_improve = 0
    unfreeze_done = False

    for epoch in range(1, num_epochs + 1):
        if epoch == 4 and not unfreeze_done:
            ctx.optimizer, ctx.scheduler = _unfreeze_backbone(
                model, ctx.optimizer.param_groups[0]["lr"] * 10, 0.0
            )
            unfreeze_done = True

        train_loss, train_acc, val_loss, val_acc, val_f1 = _run_epoch(
            epoch, num_epochs, ctx
        )
        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "lr": ctx.optimizer.param_groups[0]["lr"],
            },
            step=epoch,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), artifact_ctx.best_ckpt)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    _log_and_register(model, artifact_ctx)


# ---------- main ----------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B0 on PlantVillage"
    )
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument(
        "--mlflow-uri", type=str, default=os.getenv("MLFLOW_TRACKING_URI")
    )
    parser.add_argument("--experiment-name", type=str, default="crop-disease-detection")
    parser.add_argument(
        "--model-name", type=str, default="crop-disease-efficientnet-b0"
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="/tmp/crop_disease_model")
    parser.add_argument(
        "--model-bucket",
        type=str,
        default=os.getenv("MODEL_BUCKET"),
        help="S3 bucket for uploading reference data (optional)",
    )
    return parser.parse_args()


def main() -> None:  # pylint: disable=too-many-locals
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir, args.batch_size, args.num_workers
    )

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_params(
            {
                "model": "efficientnet_b0",
                "pretrained": "imagenet",
                "num_classes": metadata["num_classes"],
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "device": str(DEVICE),
                "leaf_aware_split": metadata.get("leaf_aware_split", False),
            }
        )

        model = build_model(metadata["num_classes"], freeze_backbone=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = _make_optimizer(model, args.lr, args.weight_decay)
        best_ckpt = str(output_dir / "best_model.pt")

        epoch_ctx = _EpochContext(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            _make_scheduler(optimizer),
        )
        artifact_ctx = _ArtifactContext(
            test_loader=test_loader,
            criterion=criterion,
            data_dir=data_dir,
            output_dir=output_dir,
            best_ckpt=best_ckpt,
            model_name=args.model_name,
            class_names=metadata["class_names"],
            model_bucket=args.model_bucket,
        )
        _run_training_loop(
            model, epoch_ctx, artifact_ctx, args.epochs, args.early_stop_patience
        )

    print(f"RUN_ID={run_id}")
    logger.info("Training complete. RUN_ID=%s", run_id)


if __name__ == "__main__":
    main()
