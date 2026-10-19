# Road-Scene Semantic Segmentation: U-Net vs SegFormer

Computer Vision course project comparing CNN-based **U-Net (ResNet-34)** with Transformer-based **SegFormer-B0** on road-scene semantic segmentation. Results below are **validation-set results as of 20 October 2026**; the held-out test set has **not** been evaluated.

## Research question

How do U-Net and SegFormer-B0 compare on a small dataset in segmentation accuracy, per-class IoU, inference latency, model size, and qualitative failure cases? Can pedestrian-weighted loss improve a weaker Transformer result, and does simple augmentation help U-Net?

## Dataset and pipeline

- **CamVid**: 701 RGB images, fixed **367 train / 101 validation / 233 test** split.
- Five remapped target classes: **road (0), vehicle (1), building (2), pedestrian (3), sky (4)**.
- All other source classes are **void (255)**, ignored by the loss and reported metrics. Sidewalk is void; lane markings are road; bicyclists are pedestrians.
- Images resized to **480 × 360** (width × height), normalized with ImageNet statistics; segmentation masks use **nearest-neighbor interpolation**.
- Normal training augmentation: horizontal flip (`p=0.5`) plus brightness/contrast jitter (limits ±0.2, `p=0.5`); validation has no random augmentation.
- CamVid is downloaded through `kagglehub` or may be placed in `data/CamVid`.

## Models and training

| Experiment | Model / change | Training setup | Checkpoint |
|---|---|---|---|
| U-Net baseline | U-Net, ImageNet-pretrained ResNet-34 encoder | 40 epochs; batch 8; AdamW LR 3e-4; cosine scheduler; cross-entropy | `unet_baseline.pt` |
| SegFormer-B0 | ADE20K-pretrained segmentation model, 5-class head | 40 epochs; microbatch 4 × accumulation 2; AdamW LR 6e-5; cross-entropy | `checkpoints/segformer_b0_best/` |
| Weighted SegFormer | Pedestrian weight 2.0, others 1.0 | Same overall fine-tuning recipe; weighted cross-entropy | `checkpoints/segformer_b0_weighted_best/` |
| U-Net no augmentation | Same U-Net setup, **no random flip/jitter** | 40 epochs; resize/normalize only | `checkpoints/unet_no_aug_best.pt` |

All loss calculations ignore pixels labeled 255. Checkpoint selection uses best **validation mIoU**, not test scores. Optimizers, learning rates and initialization differ across architectures, so the U-Net-vs-SegFormer study is a practical model comparison, **not** an architecture-only controlled ablation. The U-Net augmentation comparison is exploratory because baseline and ablation seeds were not fully matched.

## Validation results (101 images)

| Metric | U-Net + aug | U-Net no aug | SegFormer | Weighted SegFormer |
|---|---:|---:|---:|---:|
| Road IoU | 0.9912 | Not recorded | 0.9866 | 0.9870 |
| Vehicle IoU | 0.8963 | Not recorded | 0.8324 | 0.8356 |
| Building IoU | 0.9701 | Not recorded | 0.9525 | 0.9535 |
| Pedestrian IoU | 0.7037 | 0.7095* | 0.6036 | 0.6380 |
| Sky IoU | 0.9882 | Not recorded | 0.9790 | 0.9793 |
| **mIoU** | **0.9099** | **0.9019** | **0.8708** | **0.8787** |
| Pixel accuracy | 0.9860 | Not recorded | 0.9779 | 0.9785 |

*The no-augmentation pedestrian value 0.7095 comes from its **best-mIoU epoch 27** training log, not a standalone reevaluation. The remaining per-class values and pixel accuracy were not supplied; do not invent them.*

- With augmentation, U-Net achieved **+0.0080 mIoU (+0.80 percentage points)** compared with the no-augmentation U-Net run.
- Weighted SegFormer achieved **+0.0079 mIoU (+0.79 pp)** and **+0.0344 pedestrian IoU (+3.44 pp)** versus unweighted SegFormer; the best weighted checkpoint was selected at epoch **38**.
- U-Net with augmentation retains the highest validation mIoU, **0.9099**.

## GPU inference benchmark

Measured on **NVIDIA GeForce RTX 4060 Laptop GPU**, **PyTorch 2.11.0+cu128**, FP32, one **480×360** image per pass, **20 warm-up** passes and **100 timed** passes. GPU forward, full-resolution logits and argmax are timed; dataset I/O and preprocessing are excluded. FPS is `1000 / mean_latency_ms`, *not* end-to-end video throughput.

| Metric | U-Net | SegFormer-B0 |
|---|---:|---:|
| Mean latency | 13.158 ms | **10.627 ms** |
| Median latency | 12.386 ms | **9.196 ms** |
| 95th percentile | 18.581 ms | **15.234 ms** |
| Derived FPS | 76.0 | **94.1** |
| Parameters | 24.437 M | **3.715 M** |
| Peak allocated GPU memory | **196.1 MiB** | 239.4 MiB |

SegFormer was ~**19.2% lower latency** and ~**23.8% higher derived FPS**, despite higher peak allocated GPU memory. Weighted SegFormer was **not separately benchmarked**.

## Qualitative validation analysis

Predictions are compared against the same held-out **validation** frames. **Void ground-truth regions are hidden** in overlays; they do not contribute to IoU. The five-panel figures in `outputs/weighted_comparison/` show RGB, ground truth, U-Net, SegFormer, and weighted SegFormer.

| Validation image | U-Net pedestrian IoU | Original SegFormer | Weighted SegFormer | Finding |
|---|---:|---:|---:|---|
| `#042` (`0016E5_05640.png`) | 0.766 | 0.359 | **0.557** | Weighted loss markedly improves pedestrian overlap |
| `#043` (`0016E5_05670.png`) | 0.633 | 0.307 | **0.336** | Modest recovery; adjacent sequence to #042 |
| `#063` (`0016E5_08051.png`) | 0.884 | 0.774 | **0.780** | Both Transformers still trail U-Net |
| `#074` (`Seq05VD_f00180.png`) | 0.681 | 0.000 | **0.000** | Failure case remains under weighted loss |
| `#085` (`Seq05VD_f02310.png`) | 0.657 | 0.540 | **0.565** | Small improvement; vehicle IoU also rises 0.509 → 0.552 |

Per-image IoU is **not** the same as dataset-wide class IoU. A per-image IoU can be undefined if a class is absent from both labels and predictions. The images support a descriptive error analysis but do not prove that attention or decoder design caused specific errors.

## Local setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install transformers safetensors matplotlib
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Use a CUDA-enabled PyTorch build compatible with your NVIDIA driver (see [official installer](https://pytorch.org/get-started/locally/)). Requirements include PyTorch, torchvision, segmentation-models-pytorch, albumentations, opencv-python, numpy and kagglehub. If a Cyrillic Windows username breaks `cv2.imread`, use a Unicode-safe `np.fromfile` + `cv2.imdecode` reader or store CamVid at an ASCII-only path.

## Reproduce experiments

Run these **from the repository root** (scripts under `src/`):

```powershell
# U-Net baseline: train (overwrites checkpoint) / validate
python src\train.py
python src\evaluate.py

# SegFormer: train / validate
python src\train_segformer.py
python src\evaluate_segformer.py

# Weighted SegFormer: train / validate
python src\train_segformer_weighted.py
python src\evaluate_segformer.py --checkpoint checkpoints/segformer_b0_weighted_best

# No-augmentation U-Net ablation: train
python src\train_unet_no_aug.py

# Visual comparisons (validation split)
python src\visualize_comparison.py --num-images 6
python src\visualize_weighted_comparison.py --indices 42 43 63 74 85

# CUDA timing for original U-Net and SegFormer
python src\benchmark_inference.py
```

Results/checkpoints saved in the repo root include: `unet_baseline.pt`, `checkpoints/segformer_b0_best/`, `checkpoints/segformer_b0_weighted_best/`, `checkpoints/unet_no_aug_best.pt`, `outputs/validation_comparison/`, `outputs/weighted_comparison/`, and `outputs/benchmark_inference.json`. These are generated, and are not expected in a fresh clone. Before retraining, **back up current checkpoints**. To obtain *full* no-augmentation per-class scores, evaluate `checkpoints/unet_no_aug_best.pt` with the same validation loader and confusion matrix as the baseline.

## Project structure

```text
.
├── src/
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── train_segformer.py
│   ├── evaluate_segformer.py
│   ├── train_segformer_weighted.py
│   ├── train_unet_no_aug.py
│   ├── benchmark_inference.py
│   ├── visualize_comparison.py
│   └── visualize_weighted_comparison.py
├── data/CamVid/                   # Optional local dataset (untracked)
├── checkpoints/                  # Generated trained weights
├── outputs/                      # Generated plots and benchmark JSON
├── requirements.txt
└── README.md
```

## Conclusions and limitations

The current **validation** ranking is **U-Net + aug > U-Net no aug > Weighted SegFormer > SegFormer** by mIoU. U-Net is stronger on vehicle and pedestrian IoU, while original SegFormer is faster and substantially smaller in parameter count. Weighting pedestrians helps SegFormer but does not resolve every small-object failure. Augmentation correlates with a modest overall U-Net gain in the observed runs, but matched-seed repeats are needed for stronger causal claims.

**Not yet complete:** Full no-augmentation per-class validation metrics; repeated-seed uncertainty; separate weighted-SegFormer speed benchmark; and **final held-out test evaluation**. Do not report validation scores as test results. Final test-set evaluation should occur only after model and analysis choices have been frozen.


