"""Benchmark U-Net vs SegFormer-B0 GPU inference on CamVid validation images.

Run from repository root:
    python src/benchmark_inference.py

Both models use identical 480x360 validation tensors and batch size 1.
Excludes disk loading/preprocessing/visualization. Includes SegFormer logits
upsampling and argmax to obtain a full-resolution prediction for both models.
No test data is loaded.
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation

from dataset import CamVidDataset, EVAL_TF, NUM_CLASSES, ensure_camvid


def load_models(unet_path, segformer_path, device):
    unet = smp.Unet(encoder_name="resnet34", encoder_weights=None, classes=NUM_CLASSES)
    try:
        state = torch.load(unet_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(unet_path, map_location="cpu")
    unet.load_state_dict(state)
    unet = unet.to(device).eval()
    segformer = SegformerForSemanticSegmentation.from_pretrained(segformer_path).to(device).eval()
    return {"U-Net (ResNet-34)": unet, "SegFormer-B0": segformer}


def predict(name, model, x):
    if name.startswith("U-Net"):
        logits = model(x)
    else:
        logits = model(pixel_values=x).logits
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return logits.argmax(dim=1)


def percentile(values, p):
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * p / 100
    lo, hi = int(position), min(int(position) + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (position - lo)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--unet-checkpoint", type=Path, default=Path("unet_baseline.pt"))
    p.add_argument("--segformer-checkpoint", type=Path, default=Path("checkpoints/segformer_b0_best"))
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--output", type=Path, default=Path("outputs/benchmark_inference.json"))
    args = p.parse_args()
    if not torch.cuda.is_available():
        p.error("CUDA is required for the GPU benchmark. Verify your NVIDIA PyTorch installation.")
    if not args.unet_checkpoint.is_file() or not (args.segformer_checkpoint / "config.json").is_file():
        p.error("Missing checkpoint(s). Check --unet-checkpoint and --segformer-checkpoint.")
    if args.warmup < 1 or args.iterations < 2:
        p.error("Use at least one warm-up and two timed iterations.")
    device = torch.device("cuda:0")
    print("GPU:", torch.cuda.get_device_name(device), flush=True)
    ds = CamVidDataset(ensure_camvid(), "val", EVAL_TF)
    # Preload only 10 image tensors to eliminate input file I/O from timing.
    inputs = [ds[i][0].unsqueeze(0).to(device) for i in range(min(10, len(ds)))]
    assert all(tuple(x.shape[-2:]) == (360, 480) for x in inputs)
    models = load_models(args.unet_checkpoint, args.segformer_checkpoint, device)
    rows = []
    with torch.inference_mode():
        for name, model in models.items():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            for i in range(args.warmup):
                pred = predict(name, model, inputs[i % len(inputs)])
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            durations = []
            for i in range(args.iterations):
                x = inputs[i % len(inputs)]
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                pred = predict(name, model, x)
                end.record()
                end.synchronize()
                durations.append(start.elapsed_time(end))
            mean_ms = statistics.mean(durations)
            row = {
                "model": name,
                "mean_ms": round(mean_ms, 3),
                "median_ms": round(statistics.median(durations), 3),
                "p95_ms": round(percentile(durations, 95), 3),
                "fps": round(1000 / mean_ms, 2),
                "parameters_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 3),
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 1024**2, 1),
            }
            rows.append(row)
            print(f"{name}: {row['mean_ms']:.2f} ms/image (median {row['median_ms']:.2f}, "
                  f"p95 {row['p95_ms']:.2f}), {row['fps']:.2f} FPS, "
                  f"{row['parameters_m']:.2f} M params, "
                  f"peak allocated {row['peak_allocated_mib']:.1f} MiB", flush=True)
    output = {
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "resolution_hw": [360, 480], "batch_size": 1,
        "warmup": args.warmup, "iterations": args.iterations,
        "timing_scope": "GPU forward, full-resolution logits and argmax; excludes image I/O/preprocessing",
        "data_split": "validation", "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("Saved:", args.output)
    print("Test split untouched.")


if __name__ == "__main__":
    main()
