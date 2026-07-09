"""
Multi-seed AEL High-Risk weight sensitivity experiment.

Trains DenseNet-121 for 10 epochs across HR weights {3,4,5,6,7},
with N=3 independent random seeds per configuration.
Reports mean ± std for Macro F1 and High-Risk Recall.

Same controlled protocol as ael_sensitivity.py (random init, no CLAHE)
with added seed repetitions to quantify run-to-run variance.

Crash-safe: progress is saved to checkpoints/sensitivity_multiseed_progress.json
after every (HR weight, seed) pair. Restart the script to resume — completed
runs are skipped automatically.

Usage:
    python ael_sensitivity_multiseed.py
"""
import os, json, random, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')

PROJECT_DIR   = '/Users/rahmanazizur/Desktop/GastroEndoscopy-Risk-Stratification'
DATA_SEED     = 42
SEEDS         = [42, 0, 123]
NUM_CLASSES   = 4
IMG_SIZE      = 224
BATCH_SIZE    = 32
EPOCHS        = 10
LR            = 1e-4
DROPOUT       = 0.5
HR_WEIGHTS    = [3.0, 4.0, 5.0, 6.0, 7.0]
BASE_WEIGHTS  = [1.0, 3.5, 3.0]
PROGRESS_FILE = os.path.join(PROJECT_DIR, 'checkpoints', 'sensitivity_multiseed_progress.json')

DEVICE = (torch.device('mps')  if torch.backends.mps.is_available() else
          torch.device('cuda') if torch.cuda.is_available() else
          torch.device('cpu'))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_BLANK = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))

HYPERKVASIR_MAP = {
    'cecum': 0, 'pylorus': 0, 'z-line': 0, 'retroflex-stomach': 0,
    'retroflex-rectum': 0, 'ileum': 0, 'bbps-2-3': 0,
    'esophagitis-a': 1, 'ulcerative-colitis-grade-0-1': 1,
    'ulcerative-colitis-grade-1': 1, 'hemorrhoids': 1,
    'barretts': 2, 'barretts-short-segment': 2, 'esophagitis-b-d': 2,
    'polyps': 2, 'ulcerative-colitis-grade-1-2': 2, 'ulcerative-colitis-grade-2': 2,
    'ulcerative-colitis-grade-2-3': 3, 'ulcerative-colitis-grade-3': 3,
    'dyed-lifted-polyps': 3, 'dyed-resection-margins': 3,
}
EXCLUDED = {'bbps-0-1', 'impacted-stool', 'out-of-patient', 'short-segment-barretts'}


class EndoscopyDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row['image_path']).convert('RGB')
        except Exception:
            img = _BLANK.copy()
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(row['label'], dtype=torch.long)


def load_hyperkvasir(max_normal=3000):
    rng      = random.Random(DATA_SEED)
    base_dir = os.path.join(PROJECT_DIR, 'data/HyperKvasir/labeled-images')
    normal_paths, other_rows = [], []
    for root, _, files in os.walk(base_dir):
        if root.replace(base_dir, '').count(os.sep) != 3:
            continue
        cls   = os.path.basename(root)
        label = HYPERKVASIR_MAP.get(cls)
        if cls in EXCLUDED or label is None:
            continue
        imgs = [{'image_path': os.path.join(root, f), 'label': label}
                for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        (normal_paths if label == 0 else other_rows).extend(imgs)
    if len(normal_paths) > max_normal:
        rng.shuffle(normal_paths)
        normal_paths = normal_paths[:max_normal]
    return pd.DataFrame(normal_paths + other_rows)


def build_model():
    m = models.densenet121(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(DROPOUT),
                                 nn.Linear(m.classifier.in_features, NUM_CLASSES))
    return m.to(DEVICE)


def train_one(ael_weights, hr_w, seed, train_df, val_df, sampler):
    set_seed(seed)
    model     = build_model()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(ael_weights, dtype=torch.float).to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_ldr = DataLoader(EndoscopyDataset(train_df, train_tf),
                           BATCH_SIZE, sampler=sampler, num_workers=0)
    val_ldr   = DataLoader(EndoscopyDataset(val_df, val_tf), BATCH_SIZE,
                           shuffle=False, num_workers=0)

    best_val_f1, best_state = 0.0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for imgs, labels in train_ldr:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(imgs), labels).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for imgs, labels in val_ldr:
                preds.extend(model(imgs.to(DEVICE)).argmax(1).cpu().numpy())
                trues.extend(labels.numpy())
        val_f1 = f1_score(trues, preds, average='macro', zero_division=0)
        print(f'  [w={hr_w} seed={seed}] ep {epoch}/{EPOCHS}  val_f1={val_f1:.4f}', flush=True)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def evaluate(model, test_ldr):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for imgs, labels in test_ldr:
            preds.extend(model(imgs.to(DEVICE)).argmax(1).cpu().numpy())
            trues.extend(labels.numpy())
    preds, trues = np.array(preds), np.array(trues)
    macro_f1 = float(f1_score(trues, preds, average='macro', zero_division=0))
    mask     = trues == 3
    hr_rec   = float((preds[mask] == 3).sum() / mask.sum()) if mask.sum() > 0 else 0.0
    return macro_f1, hr_rec


def save_progress(progress: dict):
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    tmp = PROGRESS_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp, PROGRESS_FILE)


def main():
    print(f'Device: {DEVICE}')

    print('Loading dataset...')
    df = load_hyperkvasir()
    if len(df) == 0:
        raise RuntimeError(f'No images found at {PROJECT_DIR}/data/HyperKvasir')

    train_df, temp   = train_test_split(df,   test_size=0.30, stratify=df['label'],   random_state=DATA_SEED)
    val_df,  test_df = train_test_split(temp, test_size=0.50, stratify=temp['label'], random_state=DATA_SEED)
    print(f'Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}')

    counts  = train_df['label'].value_counts().sort_index().values
    w_per   = 1.0 / counts
    samples = torch.tensor([w_per[l] for l in train_df['label'].values], dtype=torch.float)
    sampler = WeightedRandomSampler(samples, len(samples), replacement=True)

    test_ldr = DataLoader(EndoscopyDataset(test_df, val_tf), BATCH_SIZE,
                          shuffle=False, num_workers=0)

    # Load saved progress so completed (weight, seed) pairs are skipped on resume
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            progress = json.load(f)
        done = sum(len(v) for v in progress.values())
        print(f'Resuming — {done} seed-runs already saved.')
    else:
        progress = {}

    for w in HR_WEIGHTS:
        key = str(w)
        if key not in progress:
            progress[key] = {}

    for w in HR_WEIGHTS:
        ael = BASE_WEIGHTS + [w]
        wkey = str(w)
        print(f'\n=== HR weight = {w} | AEL = {ael} ===')
        for seed in SEEDS:
            skey = str(seed)
            if skey in progress[wkey]:
                r = progress[wkey][skey]
                print(f'  seed={seed}: SKIPPED  F1={r["f1"]:.4f}  HR={r["hr"]:.4f}')
                continue
            print(f'  Running seed={seed}...')
            model    = train_one(ael, w, seed, train_df, val_df, sampler)
            f1, hr   = evaluate(model, test_ldr)
            progress[wkey][skey] = {'f1': f1, 'hr': hr}
            save_progress(progress)          # crash-safe save after every (weight, seed)
            print(f'  seed={seed}: F1={f1:.4f}  HR={hr:.4f}  [saved]')
            del model
            if DEVICE.type == 'mps':
                torch.mps.empty_cache()

    # ── Summary ────────────────────────────────────────────────────────────────
    print('\n\n========== SENSITIVITY RESULTS (mean ± std, N=3 seeds) ==========')
    print(f'{"w3":<6} {"Macro F1":>22} {"HR Recall":>22}')
    print('-' * 52)
    results = []
    for w in HR_WEIGHTS:
        wkey = str(w)
        f1s  = [progress[wkey][str(s)]['f1'] for s in SEEDS]
        hrs  = [progress[wkey][str(s)]['hr'] for s in SEEDS]
        marker = ' ◄ (ours)' if w == 5.0 else ''
        print(f'{w:<6.1f} {np.mean(f1s):.4f} ± {np.std(f1s):.4f}   '
              f'{np.mean(hrs):.4f} ± {np.std(hrs):.4f}{marker}')
        results.append({
            'HR_weight': w,
            'F1_mean': np.mean(f1s), 'F1_std': np.std(f1s),
            'HR_mean': np.mean(hrs), 'HR_std': np.std(hrs),
        })

    print('\n--- LaTeX rows (paste into Table 11 / tab:ael_sensitivity) ---')
    for r in results:
        w    = r['HR_weight']
        bold = w == 5.0
        tag  = r' \textbf{(ours)}' if bold else ''
        row  = (f'$\\mathbf{{[1.0, 3.5, 3.0, {w:.1f}]}}${tag}'
                if bold else f'$[1.0, 3.5, 3.0, {w:.1f}]$')
        f1   = (f'$\\mathbf{{{r["F1_mean"]:.4f} \\pm {r["F1_std"]:.4f}}}$'
                if bold else f'${r["F1_mean"]:.4f} \\pm {r["F1_std"]:.4f}$')
        hr   = (f'$\\mathbf{{{r["HR_mean"]:.4f} \\pm {r["HR_std"]:.4f}}}$'
                if bold else f'${r["HR_mean"]:.4f} \\pm {r["HR_std"]:.4f}$')
        print(f'{row} & {f1} & {hr} \\\\')

    out = os.path.join(PROJECT_DIR, 'ael_sensitivity_multiseed_results.csv')
    pd.DataFrame(results).to_csv(out, index=False)
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
