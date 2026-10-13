import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from dataset import (CamVidDataset, CLASS_NAMES, NUM_CLASSES, TRAIN_TF, EVAL_TF, ensure_camvid)

EPOCHS, BATCH, LR = 40, 8, 3e-4
ROOT = ensure_camvid()


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    mat = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        valid = y != 255
        idx = y[valid] * NUM_CLASSES + pred[valid]
        mat += torch.bincount(idx, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
    inter = torch.diag(mat).float()
    union = (mat.sum(0) + mat.sum(1)).float() - inter
    return inter / union.clamp(min=1), mat


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl = DataLoader(CamVidDataset(ROOT, "train", TRAIN_TF),
                           batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(CamVidDataset(ROOT, "val", EVAL_TF), batch_size=BATCH, num_workers=2)

    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                     in_channels=3, classes=NUM_CLASSES).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item(); n += 1
        sched.step()
        iou, _ = validate(model, val_dl, device)
        miou = iou.mean().item()
        print(f"epoch {epoch:02d} | train CE {total / n:.4f} | val mIoU {miou:.4f} | "
              + " ".join(f"{c}={v:.3f}" for c, v in zip(CLASS_NAMES, iou.tolist())))
        if miou > best:
            best = miou
            torch.save(model.state_dict(), "unet_baseline.pt")
    print(f"best val mIoU: {best:.4f} (checkpoint: unet_baseline.pt)")


if __name__ == "__main__":
    main()