"""Visualize U-Net versus SegFormer-B0 on identical CamVid validation images.

Run from repository root:
    python src/visualize_comparison.py --num-images 6

Requires existing src/dataset.py and the two trained checkpoints.
Only the validation split is loaded; the test split is never accessed.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation

from dataset import CamVidDataset, CLASS_NAMES, NUM_CLASSES, EVAL_TF, ensure_camvid

# Colors are for visualization only, not CamVid's original 32-class palette.
PALETTE = np.array([
    [115, 91, 213],  # road: purple
    [242, 121, 53],  # vehicle: orange
    [74, 165, 108],  # building: green
    [231, 70, 100],  # pedestrian: red
    [77, 163, 231],  # sky: blue
], dtype=np.uint8)


def colorize(mask):
    """Colorize the 5 classes. Void pixels (255) remain transparent in overlay."""
    result = np.full((*mask.shape, 3), 35, dtype=np.uint8)
    valid = (mask >= 0) & (mask < NUM_CLASSES)
    result[valid] = PALETTE[mask[valid]]
    return result


def overlay(image, mask, valid=None, alpha=0.55):
    """Blend predictions with RGB; optionally hide predictions at void GT pixels."""
    if valid is None:
        valid = np.ones(mask.shape, dtype=bool)
    blend = ((1 - alpha) * image.astype(np.float32) +
             alpha * colorize(mask).astype(np.float32)).astype(np.uint8)
    result = image.copy()
    result[valid] = blend[valid]
    return result


def per_image_iou(pred, target):
    valid = target != 255
    vals = []
    for cls in range(NUM_CLASSES):
        gt_cls = (target == cls) & valid
        pred_cls = (pred == cls) & valid
        union = (gt_cls | pred_cls).sum()
        vals.append(float((gt_cls & pred_cls).sum() / union) if union else float("nan"))
    return vals


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--unet-checkpoint", type=Path, default=Path("unet_baseline.pt"))
    p.add_argument("--segformer-checkpoint", type=Path, default=Path("checkpoints/segformer_b0_best"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/validation_comparison"))
    p.add_argument("--num-images", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--indices", type=int, nargs="*", help="Optional validation indices, 0-based")
    args = p.parse_args()
    if not args.unet_checkpoint.is_file():
        p.error(f"U-Net checkpoint missing: {args.unet_checkpoint}")
    if not (args.segformer_checkpoint / "config.json").is_file():
        p.error(f"SegFormer checkpoint missing: {args.segformer_checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    dataset = CamVidDataset(ensure_camvid(), "val", EVAL_TF)
    if args.indices is not None and len(args.indices) > 0:
        chosen = args.indices
    else:
        rng = np.random.default_rng(args.seed)
        chosen = sorted(rng.choice(len(dataset), size=min(args.num_images, len(dataset)), replace=False).tolist())
    if any(i < 0 or i >= len(dataset) for i in chosen):
        p.error(f"Indices must be in range 0..{len(dataset) - 1}")

    unet = smp.Unet(encoder_name="resnet34", encoder_weights=None, classes=NUM_CLASSES)
    try:
        checkpoint = torch.load(args.unet_checkpoint, map_location="cpu", weights_only=True)
    except TypeError:  # for older torch versions
        checkpoint = torch.load(args.unet_checkpoint, map_location="cpu")
    unet.load_state_dict(checkpoint)
    unet = unet.to(device).eval()
    segformer = SegformerForSemanticSegmentation.from_pretrained(args.segformer_checkpoint).to(device).eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Albumentations A.Normalize() default matches ImageNet mean/std.
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    for i in chosen:
        tensor, gt_tensor = dataset[i]
        gt = gt_tensor.numpy()
        rgb = np.clip((tensor.permute(1, 2, 0).numpy() * std + mean) * 255, 0, 255).astype(np.uint8)
        batch = tensor.unsqueeze(0).to(device)
        with torch.inference_mode():
            unet_pred = unet(batch).argmax(dim=1)[0].cpu().numpy()
            sf_logits = segformer(pixel_values=batch).logits
            sf_logits = F.interpolate(sf_logits, size=gt.shape, mode="bilinear", align_corners=False)
            sf_pred = sf_logits.argmax(dim=1)[0].cpu().numpy()

        # Predictions on void pixels are not scored; suppress their overlays for fair visual reading.
        scored_pixels = gt != 255
        fig, axes = plt.subplots(1, 4, figsize=(19, 5), constrained_layout=True)
        axes[0].imshow(rgb)
        axes[0].set_title("RGB input")
        axes[1].imshow(overlay(rgb, gt, scored_pixels))
        axes[1].set_title("Ground truth (void hidden)")
        axes[2].imshow(overlay(rgb, unet_pred, scored_pixels))
        axes[2].set_title("U-Net prediction")
        axes[3].imshow(overlay(rgb, sf_pred, scored_pixels))
        axes[3].set_title("SegFormer-B0 prediction")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"Validation image #{i}: {dataset.ids[i]}", fontsize=14)
        save_path = args.output_dir / f"val_{i:03d}.png"
        fig.savefig(save_path, dpi=160)
        plt.close(fig)
        ui = per_image_iou(unet_pred, gt)
        si = per_image_iou(sf_pred, gt)
        print(f"Saved {save_path}")
        print("  IoU (undefined if class absent in both GT/pred):")
        for name, a, b in zip(CLASS_NAMES, ui, si):
            print(f"    {name:<10} U-Net={a:.3f}  SegFormer={b:.3f}")
    print(f"Done: {len(chosen)} validation comparison image(s). Test split untouched.")


if __name__ == "__main__":
    main()
