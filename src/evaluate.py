import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from dataset import CamVidDataset, CLASS_NAMES, NUM_CLASSES, EVAL_TF
from train import validate, ROOT


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_dl = DataLoader(CamVidDataset(ROOT, "val", EVAL_TF), batch_size=8)
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, classes=NUM_CLASSES)
    model.load_state_dict(torch.load("unet_baseline.pt", map_location=device))
    model.to(device)
    iou, mat = validate(model, val_dl, device)
    print("Per-class IoU (validation):")
    for name, v in zip(CLASS_NAMES, iou.tolist()):
        print(f"  {name:<10} {v:.4f}")
    print(f"mIoU: {iou.mean().item():.4f}")
    print(f"pixel accuracy (valid px): "
          f"{(mat.diag().sum() / mat.sum()).item():.4f}")


if __name__ == "__main__":
    main()