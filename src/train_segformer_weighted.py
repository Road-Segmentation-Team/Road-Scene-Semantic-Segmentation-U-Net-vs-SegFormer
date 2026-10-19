"""Controlled experiment: SegFormer-B0 with pedestrian-weighted cross entropy.

Run from repository root: python src/train_segformer_weighted.py
Uses the existing src/dataset.py (including any Windows Unicode-path fix).
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from dataset import CamVidDataset, CLASS_NAMES, NUM_CLASSES, TRAIN_TF, EVAL_TF, ensure_camvid

BASE_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
CHECKPOINT_DIR = Path("checkpoints/segformer_b0_weighted_best")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def forward_logits(model, images, target_shape):
    # SegFormer predicts low-resolution logits; resize before comparing masks.
    logits = model(pixel_values=images).logits
    return F.interpolate(logits, size=target_shape, mode="bilinear", align_corners=False)


@torch.no_grad()
def validate_segformer(model, loader, device):
    model.eval()
    mat = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        pred = forward_logits(model, images, targets.shape[-2:]).argmax(dim=1)
        valid = targets != 255
        idx = targets[valid] * NUM_CLASSES + pred[valid]
        mat += torch.bincount(idx, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
    intersection = mat.diag().float()
    union = (mat.sum(0) + mat.sum(1)).float() - intersection
    iou = intersection / union.clamp(min=1)
    return iou, mat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Microbatch size; lower if CUDA runs out of memory")
    parser.add_argument("--accumulation-steps", type=int, default=2,
                        help="Effective batch = batch-size times accumulation-steps")
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="0 is safest on Windows; tune after first successful run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pedestrian-weight", type=float, default=2.0,
                        help="Weight for pedestrian class (index 3); all others stay at 1.0")
    args = parser.parse_args()
    if args.pedestrian_weight <= 0:
        parser.error("pedestrian-weight must be positive")
    if args.batch_size < 1 or args.accumulation_steps < 1 or args.epochs < 1:
        parser.error("epochs, batch-size, and accumulation-steps must be positive")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; effective batch: {args.batch_size * args.accumulation_steps}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: no CUDA GPU detected; training on CPU may be slow")

    root = ensure_camvid()  # run only within main (avoids Windows worker repeated imports)
    train_loader = DataLoader(CamVidDataset(root, "train", TRAIN_TF),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(CamVidDataset(root, "val", EVAL_TF),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    id2label = {i: name for i, name in enumerate(CLASS_NAMES)}
    label2id = {name: i for i, name in enumerate(CLASS_NAMES)}
    print(f"Loading pretrained model: {BASE_MODEL}")
    model = SegformerForSemanticSegmentation.from_pretrained(
        BASE_MODEL, num_labels=NUM_CLASSES, id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)
    print("A warning that the segmentation classifier was newly initialized is expected (150 -> 5 classes).")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # Isolate the effect of emphasizing pedestrian pixels.
    class_weights = torch.ones(NUM_CLASSES, dtype=torch.float32, device=device)
    class_weights[CLASS_NAMES.index("pedestrian")] = args.pedestrian_weight
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    print(f"Weighted CE class weights: {dict(zip(CLASS_NAMES, class_weights.tolist()))}")
    best = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        count = 0
        for batch_index, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            logits = forward_logits(model, images, targets.shape[-2:])
            loss = criterion(logits, targets)
            # Divide by actual group size (including a possibly shorter final group).
            remaining = len(train_loader) - (batch_index // args.accumulation_steps) * args.accumulation_steps
            group_size = min(args.accumulation_steps, remaining)
            (loss / group_size).backward()
            running_loss += loss.item()
            count += 1
            if (batch_index + 1) % args.accumulation_steps == 0 or batch_index + 1 == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        iou, matrix = validate_segformer(model, val_loader, device)
        miou = iou.mean().item()
        pixel_acc = (matrix.diag().sum().float() / matrix.sum().clamp(min=1)).item()
        metrics = {"epoch": epoch, "train_weighted_ce": running_loss / max(count, 1),
                   "pedestrian_weight": args.pedestrian_weight,
                   "val_miou": miou, "val_pixel_accuracy": pixel_acc,
                   "per_class_iou": dict(zip(CLASS_NAMES, iou.cpu().tolist()))}
        history.append(metrics)
        print(f"epoch {epoch:02d} | train weighted CE {metrics['train_weighted_ce']:.4f} | "
              f"val mIoU {miou:.4f} | "
              + " ".join(f"{k}={v:.3f}" for k, v in metrics["per_class_iou"].items()), flush=True)
        if miou > best:
            best = miou
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(CHECKPOINT_DIR, safe_serialization=True)
            (CHECKPOINT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            print(f"  Saved new best checkpoint to {CHECKPOINT_DIR}")
        CHECKPOINT_DIR.parent.mkdir(parents=True, exist_ok=True)
        (CHECKPOINT_DIR.parent / "segformer_b0_weighted_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8")

    print(f"Best validation mIoU: {best:.4f}")
    print("Test split not used.")


if __name__ == "__main__":
    main()
