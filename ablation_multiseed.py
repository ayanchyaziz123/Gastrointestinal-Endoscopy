"""
Multi-seed AEL ablation: AEL vs Cross-Entropy vs Focal Loss on DenseNet-121.

Runs N=3 independent seeds per loss configuration and reports mean ± std.
Results are printed as LaTeX rows ready to paste into Table 12 (tab:ablation).

Crash-safe: progress is saved to checkpoints/ablation_multiseed_progress.json
after every seed. Restart the script to resume — completed runs are skipped.

Usage:
    python ablation_multiseed.py
"""
import os, json, random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, recall_score

from src.config import DEVICE, PROJECT_DIR
from src.dataset import load_hyperkvasir, split_dataset, make_loaders
from src.models import build_densenet121
from src.loss import AsymmetricEndoscopyLoss, FocalLoss

SEEDS         = [42, 0, 123]
EPOCHS        = 10
PROGRESS_FILE = os.path.join(PROJECT_DIR, 'checkpoints', 'ablation_multiseed_progress.json')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one(loss_fn, train_ldr, test_ldr, seed: int):
    set_seed(seed)
    model = build_densenet121().to(DEVICE)
    opt   = AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sch   = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for imgs, labels in train_ldr:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(imgs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        print(f'  seed={seed}  epoch={epoch}/{EPOCHS}', flush=True)

    model.eval()
    preds_all, trues_all = [], []
    with torch.no_grad():
        for imgs, labels in test_ldr:
            preds_all.extend(model(imgs.to(DEVICE)).argmax(1).cpu().numpy())
            trues_all.extend(labels.numpy())

    macro_f1  = float(f1_score(trues_all, preds_all, average='macro', zero_division=0))
    hr_recall = float(recall_score(
        [1 if t == 3 else 0 for t in trues_all],
        [1 if p == 3 else 0 for p in preds_all],
        zero_division=0))

    if DEVICE.type == 'mps':
        torch.mps.empty_cache()
    return macro_f1, hr_recall


def save_progress(progress: dict):
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    tmp = PROGRESS_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp, PROGRESS_FILE)


def main():
    data_root = os.path.join(PROJECT_DIR, 'data', 'HyperKvasir')
    df        = load_hyperkvasir(data_root)
    if len(df) == 0:
        raise RuntimeError(f'No images found at {data_root}')

    train_df, val_df, test_df     = split_dataset(df)
    train_ldr, val_ldr, test_ldr  = make_loaders(train_df, val_df, test_df)

    loss_configs = {
        'Cross-Entropy (uniform)':             nn.CrossEntropyLoss(),
        'Focal Loss ($\\gamma=2$)':            FocalLoss(gamma=2.0),
        'AEL $[1.0, 3.5, 3.0, 5.0]$ (Ours)': AsymmetricEndoscopyLoss(),
    }

    # Load saved progress so completed seed-runs are skipped on resume
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            progress = json.load(f)
        done = sum(len(v) for v in progress.values())
        print(f'Resuming — {done} seed-runs already saved.')
    else:
        progress = {}

    for name in loss_configs:
        if name not in progress:
            progress[name] = {}

    for name, loss_fn in loss_configs.items():
        print(f'\n>>> {name}')
        for seed in SEEDS:
            key = str(seed)
            if key in progress[name]:
                r = progress[name][key]
                print(f'  seed={seed}: SKIPPED  F1={r["f1"]:.4f}  HR={r["hr"]:.4f}')
                continue
            print(f'  Running seed={seed}...')
            f1, hr = run_one(loss_fn, train_ldr, test_ldr, seed)
            progress[name][key] = {'f1': f1, 'hr': hr}
            save_progress(progress)          # crash-safe save after every seed
            print(f'  seed={seed}: F1={f1:.4f}  HR={hr:.4f}  [saved]')

    # ── Summary ────────────────────────────────────────────────────────────────
    print('\n\n========== ABLATION RESULTS (mean ± std, N=3 seeds) ==========')
    print(f'{"Loss":<44} {"Macro F1":>18} {"HR Recall":>20}')
    print('-' * 84)
    for name in loss_configs:
        f1s = [progress[name][str(s)]['f1'] for s in SEEDS]
        hrs = [progress[name][str(s)]['hr'] for s in SEEDS]
        print(f'{name:<44} {np.mean(f1s):.4f} ± {np.std(f1s):.4f}   '
              f'{np.mean(hrs):.4f} ± {np.std(hrs):.4f}')

    print('\n--- LaTeX rows (paste into Table 12 / tab:ablation) ---')
    for name in loss_configs:
        f1s    = [progress[name][str(s)]['f1'] for s in SEEDS]
        hrs    = [progress[name][str(s)]['hr'] for s in SEEDS]
        bold   = name.startswith('AEL')
        f1_str = (f'\\mathbf{{{np.mean(f1s):.4f} \\pm {np.std(f1s):.4f}}}'
                  if bold else f'{np.mean(f1s):.4f} \\pm {np.std(f1s):.4f}')
        hr_str = f'{np.mean(hrs):.4f} \\pm {np.std(hrs):.4f}'
        print(f'{name} & ${f1_str}$ & ${hr_str}$ \\\\')

    # Also save full JSON results
    out = os.path.join(PROJECT_DIR, 'ablation_multiseed_results.json')
    with open(out, 'w') as f:
        json.dump(progress, f, indent=2)
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
