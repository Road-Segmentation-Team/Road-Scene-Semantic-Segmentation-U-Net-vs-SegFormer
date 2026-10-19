import csv
import os

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

NUM_CLASSES = 5
CLASS_NAMES = ["road", "vehicle", "building", "pedestrian", "sky"]
VOID_IDX = 255  # ignored in loss and metrics

# Kaggle: CamVid, 701 images (367 train / 101 val / 233 test).
# Public dataset -> no Kaggle account or API token required.
KAGGLE_DATASET = "carlolepelaars/camvid"
LOCAL_ROOT = "data/CamVid"

# Remapping of the 32 original CamVid classes to the 5 target classes.
# Everything not listed here becomes void (255).
NAME_TO_TARGET = {
    # road (+ lane markings / shoulder)
    "Road": 0, "LaneMkgsDriv": 0, "LaneMkgsNonDriv": 0, "RoadShoulder": 0,
    # vehicles
    "Car": 1, "SUVPickupTruck": 1, "Truck_Bus": 1, "Train": 1,
    "OtherMoving": 1, "MotorcycleScooter": 1,
    # buildings / walls / structures
    "Building": 2, "Wall": 2, "Bridge": 2, "Tunnel": 2, "Archway": 2,
    # pedestrians (incl. children, prams, bicyclists)
    "Pedestrian": 3, "Child": 3, "CartLuggagePram": 3, "Bicyclist": 3,
    # sky
    "Sky": 4,
}

TRAIN_TF = A.Compose([
    A.Resize(360, 480),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.Normalize(),
    ToTensorV2(),
])

EVAL_TF = A.Compose([
    A.Resize(360, 480),
    A.Normalize(),
    ToTensorV2(),
])


def ensure_camvid(local_root=LOCAL_ROOT):
    if os.path.isfile(os.path.join(local_root, "class_dict.csv")):
        return local_root

    import kagglehub

    print(f"Downloading '{KAGGLE_DATASET}' from Kaggle (one-time, ~0.5 GB)...")
    root = kagglehub.dataset_download(KAGGLE_DATASET)

    if not os.path.isfile(os.path.join(root, "class_dict.csv")):
        for dirpath, _, files in os.walk(root):
            if "class_dict.csv" in files:
                return dirpath
        raise FileNotFoundError(f"class_dict.csv not found under {root}")
    return root


def build_color_lut(class_dict_path):
    lut = np.full(2 ** 24, VOID_IDX, dtype=np.uint8)
    with open(class_dict_path, newline="") as f:
        for row in csv.DictReader(f):
            r, g, b = int(row["r"]), int(row["g"]), int(row["b"])
            lut[(r << 16) | (g << 8) | b] = NAME_TO_TARGET.get(row["name"], VOID_IDX)
    return lut


class CamVidDataset(Dataset):
    def __init__(self, root, split, transform):
        assert split in ("train", "val", "test")
        root = ensure_camvid(root)  # download on first use
        self.img_dir = os.path.join(root, split)
        self.mask_dir = os.path.join(root, split + "_labels")
        self.transform = transform
        self.lut = build_color_lut(os.path.join(root, "class_dict.csv"))
        self.ids = sorted(f for f in os.listdir(self.img_dir) if f.endswith(".png"))

    def _mask_path(self, name):
        stem = os.path.splitext(name)[0]
        for suffix in ("_L.png", ".png"):
            p = os.path.join(self.mask_dir, stem + suffix)
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(f"no mask for {name} in {self.mask_dir}")

    def __len__(self):
        return len(self.ids)


    def __getitem__(self, i):
        name = self.ids[i]

        # Read images safely from Unicode paths on Windows.
        image_path = os.path.join(self.img_dir, name)
        mask_path = self._mask_path(name)

        image_bytes = np.fromfile(image_path, dtype=np.uint8)
        mask_bytes = np.fromfile(mask_path, dtype=np.uint8)

        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        mask = cv2.imdecode(mask_bytes, cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"Failed to decode image: {image_path}")
        if mask is None:
            raise RuntimeError(f"Failed to decode mask: {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        packed = (
            (mask[..., 0].astype(np.int32) << 16)
            | (mask[..., 1].astype(np.int32) << 8)
            | mask[..., 2].astype(np.int32)
        )

        target = self.lut[packed]
        out = self.transform(image=image, mask=target)

        return out["image"], out["mask"].long()
