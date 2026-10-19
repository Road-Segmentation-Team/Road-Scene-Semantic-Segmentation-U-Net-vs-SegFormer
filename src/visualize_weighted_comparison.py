"""Compare U-Net, original SegFormer and weighted SegFormer on identical validation images.
Run: python src/visualize_weighted_comparison.py --indices 42 43 63 74 85
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation
from dataset import CamVidDataset, NUM_CLASSES, CLASS_NAMES, EVAL_TF, ensure_camvid
from visualize_comparison import overlay, per_image_iou


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--indices', nargs='*', type=int, default=[42, 43, 63, 74, 85])
    p.add_argument('--unet-checkpoint', type=Path, default=Path('unet_baseline.pt'))
    p.add_argument('--segformer-checkpoint', type=Path, default=Path('checkpoints/segformer_b0_best'))
    p.add_argument('--weighted-checkpoint', type=Path, default=Path('checkpoints/segformer_b0_weighted_best'))
    p.add_argument('--output-dir', type=Path, default=Path('outputs/weighted_comparison'))
    args = p.parse_args()
    for checkpoint in [args.segformer_checkpoint, args.weighted_checkpoint]:
        if not (checkpoint/'config.json').is_file(): p.error(f'Missing checkpoint: {checkpoint}')
    if not args.unet_checkpoint.is_file(): p.error(f'Missing checkpoint: {args.unet_checkpoint}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = CamVidDataset(ensure_camvid(), 'val', EVAL_TF)
    if any(i < 0 or i >= len(data) for i in args.indices): p.error('Out-of-range validation index')
    unet = smp.Unet(encoder_name='resnet34', encoder_weights=None, classes=NUM_CLASSES)
    unet.load_state_dict(torch.load(args.unet_checkpoint, map_location='cpu', weights_only=True))
    models = [unet.to(device).eval(),
              SegformerForSemanticSegmentation.from_pretrained(args.segformer_checkpoint).to(device).eval(),
              SegformerForSemanticSegmentation.from_pretrained(args.weighted_checkpoint).to(device).eval()]
    names = ['U-Net', 'SegFormer-B0', 'Weighted SegFormer-B0']
    mean = np.array([.485,.456,.406], dtype=np.float32)
    std = np.array([.229,.224,.225], dtype=np.float32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for idx in args.indices:
        tensor, target = data[idx]
        gt = target.numpy()
        rgb = np.clip((tensor.permute(1,2,0).numpy()*std+mean)*255, 0, 255).astype(np.uint8)
        x = tensor.unsqueeze(0).to(device)
        predictions=[]
        with torch.inference_mode():
            for n, model in enumerate(models):
                logits = model(x) if n == 0 else model(pixel_values=x).logits
                if logits.shape[-2:] != gt.shape:
                    logits = F.interpolate(logits, size=gt.shape, mode='bilinear', align_corners=False)
                predictions.append(logits.argmax(1)[0].cpu().numpy())
        valid = gt != 255
        labels = ['RGB input', 'Ground truth (void hidden)'] + names
        panels = [rgb, overlay(rgb, gt, valid)] + [overlay(rgb, pred, valid) for pred in predictions]
        fig, axes = plt.subplots(1, 5, figsize=(24,5), constrained_layout=True)
        for ax, title, panel in zip(axes, labels, panels):
            ax.imshow(panel); ax.set_title(title); ax.axis('off')
        fig.suptitle(f'Validation #{idx}: {data.ids[idx]}')
        dest = args.output_dir/f'val_{idx:03d}.png'
        fig.savefig(dest, dpi=140); plt.close(fig)
        print('Saved', dest)
        for cls, values in zip(CLASS_NAMES, zip(*(per_image_iou(pred,gt) for pred in predictions))):
            print(f'  {cls:<10} ' + ' '.join(f'{name}={v:.3f}' for name,v in zip(names,values)))
    print('Test split untouched.')

if __name__ == '__main__':
    main()
