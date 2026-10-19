# Road-Scene Semantic Segmentation: U-Net vs SegFormer
## Milestone 2 — Second Approach and Initial Results
**Date:** 19 October 2026  
**Dataset:** CamVid (367 train / 101 validation / 233 test)  
**Status:** Initial validation comparison and six qualitative visualizations completed; inference benchmark and a pedestrian-weighted loss follow-up completed; U-Net augmentation ablation pending.

### 1. New approach: SegFormer-B0

The second approach fine-tunes **SegFormer-B0**, a lightweight hierarchical Transformer architecture for semantic segmentation, on the same five CamVid target classes as the U-Net baseline: **road, vehicle, building, pedestrian, and sky**. The model is initialized using the pretrained checkpoint `nvidia/segformer-b0-finetuned-ade-512-512`; its segmentation classifier is adapted from the source dataset's 150 categories to five output categories. SegFormer predicts class logits at a reduced spatial resolution, which are bilinearly upsampled to the ground-truth mask resolution for loss computation and evaluation.

### 2. Motivation

The research question is whether an attention-based segmentation model can outperform a CNN-based U-Net on a small road-scene dataset. SegFormer combines multi-scale hierarchical representations with attention, potentially using wider scene context to distinguish ambiguous objects such as small pedestrians and partially occluded vehicles. Since the baseline's weakest category was **pedestrian**, this class was of particular interest. These are **hypotheses**, not guaranteed improvements.

### 3. Implementation and experimental setup

Both models use the same CamVid fixed splits, label remapping (32 source categories to five target classes plus void index `255`), input resolution (**480 × 360**), normalization, and validation metric. Void pixels are excluded from cross-entropy and IoU. The **test split remains untouched**.

| Property | U-Net baseline | SegFormer-B0 |
|---|---|---|
| Architecture | CNN encoder–decoder with skip connections | Hierarchical Transformer encoder with segmentation decoder |
| Initialization | ImageNet-pretrained ResNet-34 encoder | ADE20K-pretrained SegFormer-B0, new five-class classifier |
| Loss | Cross-entropy; ignore index 255 | Cross-entropy; ignore index 255 |
| Optimizer | AdamW | AdamW |
| Learning rate | 3 × 10⁻⁴ | 6 × 10⁻⁵ (initial training configuration) |
| Weight decay | 1 × 10⁻⁴ | 1 × 10⁻⁴ |
| LR scheduler | Cosine annealing | Cosine annealing |
| Epochs configured | 40 | 40 |
| Batch | 8 | Microbatch 4, gradient accumulation 2 (effective batch 8) |
| Augmentation | Horizontal flip; brightness/contrast jitter | Same dataset training pipeline |
| Evaluation | Five-class IoU and mIoU on validation | Identical confusion-matrix-based IoU and mIoU on validation |

The SegFormer implementation uses `SegformerForSemanticSegmentation` from Hugging Face Transformers. Its segmentation logits are resized with bilinear interpolation (`align_corners=False`) before calculating loss or selecting predicted classes. The model's best validation checkpoint is stored separately at `checkpoints/segformer_b0_best/`. Training history is recorded in `checkpoints/segformer_b0_history.json`.

**Comparison limitation:** Although the data and metrics are controlled, the models differ not only in architecture but also in pretraining task/domain, learning rate, and training details. The observed difference should not be attributed solely to CNN versus Transformer architecture.

### 4. Initial quantitative results (validation)

The following numbers are from the **local evaluation outputs** provided on 19 October 2026, using trained checkpoints. The values are not test-set results.

| Validation metric | U-Net | SegFormer-B0 | SegFormer − U-Net |
|---|---:|---:|---:|
| Road IoU | 0.9912 | 0.9866 | −0.0046 |
| Vehicle IoU | 0.8963 | 0.8324 | −0.0639 |
| Building IoU | 0.9701 | 0.9525 | −0.0176 |
| Pedestrian IoU | 0.7037 | 0.6036 | −0.1001 |
| Sky IoU | 0.9882 | 0.9790 | −0.0092 |
| **mIoU** | **0.9099** | **0.8708** | **−0.0391** |
| Pixel accuracy (valid pixels) | 0.9860 | 0.9779 | −0.0081 |

U-Net outperformed SegFormer-B0 on **all five classes** in this initial comparison, with a **3.91 percentage-point advantage in mIoU**. The largest SegFormer deficits were for **pedestrian (10.01 points)** and **vehicle (6.39 points)**. Road and sky were strong for both models. This is a meaningful experimental result, even though it is contrary to the original expectation that SegFormer might achieve higher segmentation quality.

### 5. Discussion and interpretation

Both models achieved high IoU for large, relatively homogeneous regions (road, building, sky). The weaker results for pedestrians and vehicles are consistent with the greater difficulty of small, thin, distant, or occluded objects, but **the current aggregate metrics cannot establish the reason for the gap**. Plausible contributors include dataset size, training optimization, the distinct pretrained checkpoints, and differences in segmentation resolution or inductive biases. These hypotheses require qualitative inspection or controlled follow-up experiments.

### 6. Qualitative analysis (six validation images)

Six validation frames were inspected side-by-side as RGB input, annotated ground truth, U-Net prediction, and SegFormer-B0 prediction. Colored overlays are restricted to non-void ground-truth pixels; an uncolored region is not necessarily a model omission because void pixels are intentionally hidden. Per-image IoU values come from the local visualization run, not visual estimates.

| Validation image | U-Net vs SegFormer per-image IoU | Observation |
|---|---|---|
| `#008` — `0006R0_f01050.png` | Vehicle **0.727 vs 0.682**; building **0.303 vs 0.323** | Both capture the large road and sky areas; both struggle with a small building/structure region on the right. SegFormer is slightly higher for building in this frame, although both have low building IoU. |
| `#042` — `0016E5_05640.png` | Pedestrian **0.766 vs 0.359**; vehicle **0.895 vs 0.794** | On a dark urban street, SegFormer is substantially weaker around the small pedestrian instances and has lower vehicle overlap. |
| `#043` — `0016E5_05670.png` | Pedestrian **0.633 vs 0.307**; vehicle **0.920 vs 0.850** | Similar nearby street scene shows reduced overlap for small pedestrians and vehicles in SegFormer. Frame #043 is adjacent in sequence to #042, so the two are not independent scene examples. |
| `#063` — `0016E5_08051.png` | Pedestrian **0.884 vs 0.774**; vehicle **0.966 vs 0.906** | Both recover the prominent foreground car and pedestrian/cyclist shapes, but U-Net agrees more closely with the annotations. |
| `#074` — `Seq05VD_f00180.png` | Pedestrian **0.681 vs 0.000**; vehicle **0.967 vs 0.920** | The most striking pedestrian failure among these selected frames: SegFormer has no pedestrian intersection with the ground-truth pixels in this frame. Tiny/distant pedestrians require closer cropped inspection before assigning a specific cause. |
| `#085` — `Seq05VD_f02310.png` | Vehicle **0.822 vs 0.509**; pedestrian **0.657 vs 0.540** | SegFormer shows substantially weaker vehicle overlap in the street scene with small distant cars. |

On image #008, pedestrian IoU is **0.000** for U-Net and **undefined (NaN)** for SegFormer: NaN means the pedestrian class is absent from both the ground-truth and SegFormer prediction within the evaluated region, not that SegFormer failed a present pedestrian instance. Do not average per-image NaNs as zero or confuse selected-image IoUs with the dataset-level confusion-matrix IoU.

Overall, these examples are consistent with the validation-wide deficit on pedestrian and vehicle classes. They support a descriptive conclusion about lower overlap for those classes, but **do not prove** which architectural or training difference caused the gap. Large road and sky regions are consistently similar across the two predictions. There is no quantified inference-speed result yet.

Figures are included as `submission_figures/val_008.png`, `val_042.png`, `val_043.png`, `val_063.png`, `val_074.png`, and `val_085.png`. For a concise milestone report, figures #042, #074, and #085 offer three clear examples of relevant differences; figure #008 is useful as a counterexample where SegFormer slightly exceeds U-Net on building for one frame.

### 7. Reproduction and next steps

The six validation visualizations have been generated and inspected. Next, benchmark inference time on the same hardware and input size, then consider a controlled follow-up experiment (such as adjusting the loss for rare classes) while retaining the current SegFormer checkpoint as the initial result.

**Commands (run from repository root):**

```powershell
python src/visualize_comparison.py --num-images 6
python src/evaluate.py
python src/evaluate_segformer.py
```

The test split must remain reserved for final evaluation, as required in the project plan. Qualitative findings above are based on the six supplied validation visualizations. Inference times and final test-set results have not yet been measured.

### 7. CUDA inference benchmark

Benchmarked on NVIDIA GeForce RTX 4060 Laptop GPU, PyTorch 2.11.0+cu128, FP32 batch size 1 and 480 × 360 input. The procedure used 20 warm-up passes and 100 timed iterations on the validation split. Timings include forward computation, full-resolution logits, and argmax; they exclude I/O and image preprocessing.

| Benchmark | U-Net | SegFormer-B0 |
|---|---:|---:|
| Mean latency, ms/image | 13.158 | **10.627** |
| Median latency, ms | 12.386 | **9.196** |
| P95 latency, ms | 18.581 | **15.234** |
| Derived FPS | 76.0 | **94.1** |
| Parameters, million | 24.437 | **3.715** |
| Peak allocated GPU memory, MiB | **196.1** | 239.4 |

SegFormer-B0 is 19.2% lower in mean latency while U-Net is more accurate. The benchmark-specific GPU memory allocation is higher for SegFormer even though its model has fewer parameters.

### 8. Controlled follow-up: pedestrian-weighted cross-entropy

The initial SegFormer training was repeated with weight **2.0 for pedestrian pixels** and weight **1.0 for all other classes** in cross-entropy; void 255 remained excluded. The best of 40 validation-selected epochs was epoch 38.

| Validation metric | Original SegFormer | Weighted SegFormer |
|---|---:|---:|
| Road | 0.9866 | 0.9870 |
| Vehicle | 0.8324 | 0.8356 |
| Building | 0.9525 | 0.9535 |
| Pedestrian | 0.6036 | **0.6380** |
| Sky | 0.9790 | 0.9793 |
| **mIoU** | **0.8708** | **0.8787** |
| Pixel accuracy | 0.9779 | 0.9785 |

Pedestrian IoU increased by **3.44 percentage points** and mIoU by **0.79 points** compared with the original SegFormer checkpoint. Even with the modification, U-Net remains ahead. This is single-run evidence; repetitions with matched seeds would be necessary to estimate variability. The test split was not used.

### 9. Planned next experiment

A U-Net augmentation sensitivity experiment will remove horizontal flipping and brightness/contrast jitter while retaining resizing, ImageNet normalization, architecture, optimizer, learning-rate scheduler, and 40-epoch training budget. Its checkpoint will be stored separately to preserve the existing models. Because the original U-Net run did not explicitly fix a random seed, any observed difference should be reported as exploratory rather than solely caused by augmentation.
