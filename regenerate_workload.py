"""
Regenerate workload_simulation.png with a fixed random seed so the
pie chart and paper Table 5 are guaranteed to agree.

Run once from the project root:
    python regenerate_workload.py

It prints the numbers to paste into paper.tex.
"""
import os, random
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ── reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

from src.config import DEVICE, PROJECT_DIR
from src.dataset import load_hyperkvasir, split_dataset, val_transforms
from src.models import build_efficientnet_b0
from src.uncertainty import mc_predict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── load model ─────────────────────────────────────────────────────────────────
CKPT = os.path.join(PROJECT_DIR, 'checkpoints', 'efficientnet_b0', 'best.pt')
model = build_efficientnet_b0().to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()
print(f'Loaded EfficientNet-B0 from {CKPT}')

# ── load test split ────────────────────────────────────────────────────────────
df = load_hyperkvasir(os.path.join(PROJECT_DIR, 'data', 'HyperKvasir'))
_, _, test_df = split_dataset(df)
total = len(test_df)
print(f'Test set: {total} images')

# ── run MC Dropout simulation ──────────────────────────────────────────────────
THRESHOLD = 0.75
T = 30
torch.manual_seed(SEED)   # fix seed again just before stochastic inference

mc_results = []
for _, row in tqdm(test_df.iterrows(), total=total, desc='MC Dropout'):
    img = val_transforms(Image.open(row['image_path']).convert('RGB'))
    mc_results.append((mc_predict(model, img, n_samples=T), row['label']))

auto_cleared = sum(1 for r, _ in mc_results
                   if r['confidence'] >= THRESHOLD and r['prediction'] < 2)
flagged       = total - auto_cleared
missed_hr     = sum(1 for r, lbl in mc_results
                    if r['confidence'] >= THRESHOLD and r['prediction'] < 2 and lbl == 3)

pct_auto  = 100 * auto_cleared / total
pct_flag  = 100 * flagged / total

print(f'\n=== RESULTS (paste into paper.tex) ===')
print(f'  Auto-cleared:  {auto_cleared} ({pct_auto:.1f}%)')
print(f'  Flagged:       {flagged} ({pct_flag:.1f}%)')
print(f'  Missed HR:     {missed_hr}')

# ── threshold sweep ────────────────────────────────────────────────────────────
thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
reductions = [100 * sum(1 for r, _ in mc_results
                        if r['confidence'] >= t and r['prediction'] < 2) / total
              for t in thresholds]

# ── plot ───────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

ax1.pie([auto_cleared, flagged],
        labels=[f'AI auto-cleared\n({pct_auto:.1f}%)',
                f'Endoscopist review\n({pct_flag:.1f}%)'],
        colors=['#2ECC71', '#E74C3C'], autopct='%1.1f%%', startangle=90)
ax1.set_title('Endoscopy Screening Workflow')

ax2.plot(thresholds, reductions, 'b-o', linewidth=2)
ax2.axvline(THRESHOLD, color='red', linestyle='--', label=f'τ={THRESHOLD}')
ax2.set(xlabel='Confidence Threshold τ',
        ylabel='Workload Reduction (%)',
        title='Threshold vs. Burden Reduction')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(PROJECT_DIR, 'workload_simulation.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nSaved → {out}')
print(f'\nUpdate paper.tex:')
print(f'  AI auto-cleared  →  {auto_cleared} ({pct_auto:.1f}\\,\\%)')
print(f'  Flagged          →  {flagged} ({pct_flag:.1f}\\,\\%)')
print(f'  Workload reduction → {pct_auto:.1f}\\,\\%')
