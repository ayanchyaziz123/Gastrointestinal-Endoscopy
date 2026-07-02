import os
import random
import warnings

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from PIL import ImageFile
from sklearn.model_selection import train_test_split

ImageFile.LOAD_TRUNCATED_IMAGES = True

from .config import (
    PROJECT_DIR, SEED, NUM_CLASSES, IMG_SIZE, BATCH_SIZE,
    CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE, LABEL_NAMES, RISK_LABELS, DEVICE,
)

HYPERKVASIR_MAP = {
    # Class 0 — Normal
    'cecum':             0,
    'pylorus':           0,
    'z-line':            0,
    'retroflex-stomach': 0,
    'retroflex-rectum':  0,
    'ileum':             0,
    'bbps-2-3':          0,
    # Class 1 — Inflammatory
    'esophagitis-a':                1,
    'ulcerative-colitis-grade-0-1': 1,
    'ulcerative-colitis-grade-1':   1,
    'hemorrhoids':                  1,
    # Class 2 — Pre-malignant
    'barretts':                     2,
    'barretts-short-segment':       2,
    'esophagitis-b-d':              2,
    'polyps':                       2,
    'ulcerative-colitis-grade-1-2': 2,
    'ulcerative-colitis-grade-2':   2,
    # Class 3 — High-Risk
    'ulcerative-colitis-grade-2-3': 3,
    'ulcerative-colitis-grade-3':   3,
    'dyed-lifted-polyps':           3,
    'dyed-resection-margins':       3,
}

EXCLUDED_CLASSES = {'bbps-0-1', 'impacted-stool', 'out-of-patient', 'short-segment-barretts'}

KVASIR_V2_MAP = {
    'esophagitis':            1,
    'ulcerative-colitis':     2,
    'polyps':                 2,
    'barretts':               2,
    'normal-cecum':           0,
    'normal-pylorus':         0,
    'normal-z-line':          0,
    'dyed-lifted-polyps':     3,
    'dyed-resection-margins': 3,
}

DATASET_MEAN = [0.485, 0.456, 0.406]
DATASET_STD  = [0.229, 0.224, 0.225]


class CLAHETransform:
    """CLAHE on the L-channel of LAB colour space to enhance mucosal texture."""
    def __init__(self, clip_limit=CLAHE_CLIP_LIMIT, tile_grid_size=CLAHE_TILE_SIZE):
        self.clip_limit     = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        try:
            img_np      = np.array(img.convert('RGB'))
            clahe       = cv2.createCLAHE(clipLimit=self.clip_limit,
                                          tileGridSize=self.tile_grid_size)
            lab         = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            lab[..., 0] = clahe.apply(lab[..., 0])
            img_np      = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return Image.fromarray(img_np)
        except Exception as e:
            warnings.warn(f'CLAHETransform failed ({e}); returning original image.')
            return img


train_transforms = transforms.Compose([
    CLAHETransform(),
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
])

val_transforms = transforms.Compose([
    CLAHETransform(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
])

_BLANK = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))


def load_hyperkvasir(data_dir: str, max_normal: int = 3000) -> pd.DataFrame:
    """Walk the nested HyperKvasir structure and map classes to 4-tier risk schema."""
    rng      = random.Random(SEED)
    base_dir = os.path.join(data_dir, 'labeled-images')

    if not os.path.exists(base_dir):
        print(f'[!] HyperKvasir not found at {base_dir}')
        return pd.DataFrame(columns=['image_path', 'label', 'label_name', 'risk', 'source_class'])

    normal_paths, other_rows, skipped = [], [], []

    for root, dirs, files in os.walk(base_dir):
        depth = root.replace(base_dir, '').count(os.sep)
        if depth != 3:
            continue
        class_folder = os.path.basename(root)
        if class_folder in EXCLUDED_CLASSES:
            continue
        label = HYPERKVASIR_MAP.get(class_folder)
        if label is None:
            skipped.append(class_folder)
            continue
        imgs = [os.path.join(root, f) for f in files
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        for path in imgs:
            row = {'image_path': path, 'label': label,
                   'label_name': LABEL_NAMES[label], 'risk': RISK_LABELS[label],
                   'source_class': class_folder}
            (normal_paths if label == 0 else other_rows).append(row)

    if skipped:
        print(f'[INFO] Skipped {len(skipped)} unmapped folders: {", ".join(sorted(skipped))}')

    if len(normal_paths) > max_normal:
        print(f'[INFO] Normal class capped: {len(normal_paths)} → {max_normal} images')
        rng.shuffle(normal_paths)
        normal_paths = normal_paths[:max_normal]

    df = pd.DataFrame(normal_paths + other_rows)
    if len(df) == 0:
        print('[!] No images found. Check data directory and folder structure.')
        return df

    print(f'\nHyperKvasir loaded: {len(df)} images across {df["source_class"].nunique()} classes')
    for label in range(NUM_CLASSES):
        subset = df[df['label'] == label]
        print(f'  Class {label} | {LABEL_NAMES[label]:<20}: {len(subset):>5}  ({100*len(subset)/len(df):.1f}%)')
    return df


def load_kvasir_v2(data_dir: str) -> pd.DataFrame:
    rows = []
    if not os.path.exists(data_dir):
        print(f'[!] Kvasir-v2 not found at {data_dir}')
        return pd.DataFrame()
    for class_folder in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_folder)
        if not os.path.isdir(class_dir):
            continue
        label = KVASIR_V2_MAP.get(class_folder)
        if label is None:
            continue
        for f in os.listdir(class_dir):
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            rows.append({'image_path': os.path.join(class_dir, f),
                         'label': label, 'label_name': LABEL_NAMES[label],
                         'source_class': class_folder})
    df = pd.DataFrame(rows)
    if len(df):
        print(f'Kvasir-v2: {len(df)} images')
        print(df['label_name'].value_counts().to_string())
    return df


def split_dataset(df, seed=SEED):
    train_df, temp   = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=seed)
    val_df, test_df  = train_test_split(temp, test_size=0.5, stratify=temp['label'], random_state=seed)
    print(f'Split: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}')
    return train_df, val_df, test_df


class EndoscopyDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform
        missing = [p for p in self.df['image_path'] if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f'{len(missing)} image paths not found. First: {missing[0]}')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row['image_path']).convert('RGB')
        except Exception as e:
            warnings.warn(f'Bad image skipped: {e}')
            img = _BLANK.copy()
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(row['label'], dtype=torch.long)


def make_loaders(train_df, val_df, test_df):
    counts  = train_df['label'].value_counts().sort_index().values
    weights = 1.0 / counts
    sw      = torch.tensor(weights[train_df['label'].values], dtype=torch.float64)
    sampler = WeightedRandomSampler(
        sw, num_samples=len(sw), replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    pin = DEVICE.type == 'cuda'
    train_ldr = DataLoader(EndoscopyDataset(train_df, train_transforms),
                           BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=pin)
    val_ldr   = DataLoader(EndoscopyDataset(val_df, val_transforms),
                           BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)
    test_ldr  = DataLoader(EndoscopyDataset(test_df, val_transforms),
                           BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)
    return train_ldr, val_ldr, test_ldr
