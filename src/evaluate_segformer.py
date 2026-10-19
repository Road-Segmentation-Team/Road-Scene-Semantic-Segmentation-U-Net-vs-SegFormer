"""Evaluate the best SegFormer checkpoint on CamVid validation only."""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from dataset import CamVidDataset, CLASS_NAMES, EVAL_TF, ensure_camvid
from train_segformer import validate_segformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/segformer_b0_best"))
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if not (args.checkpoint / "config.json").exists():
        parser.error(f"Checkpoint not found: {args.checkpoint} (train SegFormer first)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = ensure_camvid()
    loader = DataLoader(CamVidDataset(root, "val", EVAL_TF), batch_size=args.batch_size,
                        shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    model = SegformerForSemanticSegmentation.from_pretrained(args.checkpoint).to(device)
    iou, mat = validate_segformer(model, loader, device)
    print("Per-class IoU (validation):")
    for name, value in zip(CLASS_NAMES, iou.cpu().tolist()):
        print(f"  {name:<10} {value:.4f}")
    print(f"mIoU: {iou.mean().item():.4f}")
    print(f"pixel accuracy (valid px): {(mat.diag().sum() / mat.sum().clamp(min=1)).item():.4f}")
    print("Test split not used.")


if __name__ == "__main__":
    main()
