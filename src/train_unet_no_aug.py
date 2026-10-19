"""U-Net augmentation ablation. Run from repository root: python src/train_unet_no_aug.py.
Preserves unet_baseline.pt. Validation split only; test is untouched.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from dataset import CamVidDataset, CLASS_NAMES, NUM_CLASSES, EVAL_TF, ensure_camvid


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--checkpoint', type=Path, default=Path('checkpoints/unet_no_aug_best.pt'))
    args = p.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device, '| training without random augmentation', flush=True)
    root = ensure_camvid()
    train_dl = DataLoader(CamVidDataset(root, 'train', EVAL_TF), batch_size=args.batch_size,
                          shuffle=True, num_workers=args.num_workers, pin_memory=device.type=='cuda')
    val_dl = DataLoader(CamVidDataset(root, 'val', EVAL_TF), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, pin_memory=device.type=='cuda')
    model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet',
                     in_channels=3, classes=NUM_CLASSES).to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history, best = [], -1.
    for epoch in range(1, args.epochs+1):
        model.train()
        losses = []
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        model.eval()
        matrix = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long, device=device)
        with torch.inference_mode():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                valid = y != 255
                indices = y[valid]*NUM_CLASSES + pred[valid]
                matrix += torch.bincount(indices, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
        intersection = matrix.diag().float()
        union = matrix.sum(0).float() + matrix.sum(1).float() - intersection
        per_iou = (intersection / union.clamp(min=1)).cpu().tolist()
        miou = float(np.mean(per_iou))
        acc = float((matrix.diag().sum().float()/matrix.sum().clamp(min=1)).item())
        result = dict(epoch=epoch, train_ce=float(np.mean(losses)), val_miou=miou,
                      val_pixel_accuracy=acc, per_class_iou=dict(zip(CLASS_NAMES, per_iou)))
        history.append(result)
        print(f"epoch {epoch:02d} | loss {result['train_ce']:.4f} | val mIoU {miou:.4f} | pedestrian {per_iou[3]:.4f}", flush=True)
        if miou > best:
            best = miou
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.checkpoint)
            args.checkpoint.with_suffix('.metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        args.checkpoint.with_suffix('.history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    print('Best validation mIoU:', f'{best:.4f}', '| checkpoint:', args.checkpoint)
    print('Test split untouched.')

if __name__ == '__main__':
    main()
