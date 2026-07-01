"""AEL High-Risk weight sensitivity experiment.

Trains DenseNet-121 for 10 epochs with High-Risk weights 3-7,
keeping all other weights fixed at [1.0, 3.5, 3.0, w].
Reports Macro F1 and High-Risk Recall on the held-out test set.
"""
import os, random, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from PIL import Image
import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')

# ── Constants ────────────────────────────────────────────────────────────────
PROJECT_DIR  = '/Users/rahmanazizur/Desktop/GastroEndoscopy-Risk-Stratification'
SEED         = 42
NUM_CLASSES  = 4
IMG_SIZE     = 224
BATCH_SIZE   = 64
EPOCHS       = 10
LR           = 1e-4
DROPOUT      = 0.5
HR_WEIGHTS   = [3.0, 4.0, 5.0, 6.0, 7.0]   # High-Risk weight to sweep
BASE_WEIGHTS = [1.0, 3.5, 3.0]               # Normal, Inflammatory, Pre-malignant

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else \
         torch.device('cuda') if torch.cuda.is_available() else \
         torch.device('cpu')
print(f'Device: {DEVICE}')

# ── CLAHE ─────────────────────────────────────────────────────────────────────
class CLAHETransform:
    def __call__(self, img):
        try:
            img_np = np.array(img.convert('RGB'))
            clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab         = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            lab[..., 0] = clahe.apply(lab[..., 0])
            return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
        except Exception:
            return img

train_tf = transforms.Compose([
    CLAHETransform(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    CLAHETransform(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Dataset ───────────────────────────────────────────────────────────────────
_BLANK = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))

class EndoscopyDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df        = df.reset_index(drop=True)
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

# ── Data loading ──────────────────────────────────────────────────────────────
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

def load_hyperkvasir(max_normal=3000):
    rng      = random.Random(SEED)
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

df = load_hyperkvasir()
train_df, temp  = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=SEED)
val_df, test_df = train_test_split(temp, test_size=0.5, stratify=temp['label'], random_state=SEED)
print(f'Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}')

# WeightedRandomSampler
counts  = train_df['label'].value_counts().sort_index().values
w_per   = 1.0 / counts
samples = torch.tensor([w_per[l] for l in train_df['label'].values], dtype=torch.float)
sampler = WeightedRandomSampler(samples, len(samples), replacement=True)

test_ldr = DataLoader(EndoscopyDataset(test_df, val_tf),
                      BATCH_SIZE, shuffle=False, num_workers=0)

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    m = models.densenet121(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(DROPOUT),
        nn.Linear(m.classifier.in_features, NUM_CLASSES)
    )
    return m.to(DEVICE)

# ── Train ─────────────────────────────────────────────────────────────────────
def train_one(ael_weights):
    model     = build_model()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(ael_weights, dtype=torch.float).to(DEVICE)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_ldr = DataLoader(
        EndoscopyDataset(train_df, train_tf),
        BATCH_SIZE, sampler=sampler, num_workers=0
    )
    val_ldr = DataLoader(
        EndoscopyDataset(val_df, val_tf),
        BATCH_SIZE, shuffle=False, num_workers=0
    )

    best_val_f1, best_state = 0.0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for imgs, labels in train_ldr:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for imgs, labels in val_ldr:
                out = model(imgs.to(DEVICE))
                preds.extend(out.argmax(1).cpu().numpy())
                trues.extend(labels.numpy())
        val_f1 = f1_score(trues, preds, average='macro', zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
        print(f'  epoch {epoch:2d}/{EPOCHS}  val_macro_f1={val_f1:.4f}')

    model.load_state_dict(best_state)
    return model

# ── Evaluate ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model):
    model.eval()
    preds, trues = [], []
    for imgs, labels in test_ldr:
        out = model(imgs.to(DEVICE))
        preds.extend(out.argmax(1).cpu().numpy())
        trues.extend(labels.numpy())
    preds, trues = np.array(preds), np.array(trues)
    macro_f1  = f1_score(trues, preds, average='macro',    zero_division=0)
    hr_recall = f1_score(trues, preds, labels=[3], average=None, zero_division=0)
    # actual recall for class 3
    mask = trues == 3
    hr_rec = preds[mask].tolist().count(3) / mask.sum() if mask.sum() > 0 else 0.0
    return macro_f1, hr_rec

# ── Main sweep ────────────────────────────────────────────────────────────────
results = []
for w in HR_WEIGHTS:
    ael = BASE_WEIGHTS + [w]
    print(f'\n=== High-Risk weight = {w} | AEL = {ael} ===')
    model = train_one(ael)
    macro_f1, hr_recall = evaluate(model)
    results.append({'HR_weight': w, 'Macro_F1': macro_f1, 'HR_Recall': hr_recall})
    print(f'  >> Macro F1={macro_f1:.4f}  HR Recall={hr_recall:.4f}')
    del model
    if DEVICE.type == 'mps':
        torch.mps.empty_cache()

# ── Print table ───────────────────────────────────────────────────────────────
print('\n\n========== RESULTS ==========')
print(f"{'HR weight':<12} {'Macro F1':<12} {'HR Recall':<12}")
print('-' * 36)
for r in results:
    marker = ' ◄ (ours)' if r['HR_weight'] == 5.0 else ''
    print(f"{r['HR_weight']:<12.1f} {r['Macro_F1']:<12.4f} {r['HR_Recall']:<12.4f}{marker}")

df_res = pd.DataFrame(results)
out_path = os.path.join(PROJECT_DIR, 'ael_sensitivity_results.csv')
df_res.to_csv(out_path, index=False)
print(f'\nSaved to {out_path}')

# LaTeX table snippet
print('\n--- LaTeX table rows ---')
for r in results:
    marker = r'$\dagger$' if r['HR_weight'] == 5.0 else ''
    print(f"AEL [1.0, 3.5, 3.0, {r['HR_weight']:.1f}]{marker} & {r['Macro_F1']:.4f} & {r['HR_Recall']:.4f} \\\\")
