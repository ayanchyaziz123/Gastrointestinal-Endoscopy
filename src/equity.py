import os

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from .config import PROJECT_DIR, DEVICE, BATCH_SIZE
from .dataset import EndoscopyDataset, val_transforms
from .train import evaluate
from .loss import AsymmetricEndoscopyLoss


def compute_cdctei(f1_dict: dict) -> float:
    """CD-CTEI = 1 - σ(F1) / μ(F1) across demographic subgroups."""
    vals = np.array(list(f1_dict.values()))
    mu, sigma = vals.mean(), vals.std()
    return float(1 - sigma / mu) if mu > 0 else 0.0


def equity_analysis(model, test_df, device, group_col='age_group'):
    if group_col not in test_df.columns:
        print(f'Add {group_col} column to test_df for equity analysis.')
        print('Example: age_group=[<40, 40-60, >60]  or  sex=[M, F]')
        return

    criterion = AsymmetricEndoscopyLoss()
    results   = {}
    for group in sorted(test_df[group_col].unique()):
        sub = test_df[test_df[group_col] == group].reset_index(drop=True)
        ldr = DataLoader(EndoscopyDataset(sub, val_transforms), BATCH_SIZE, shuffle=False)
        f1, _, _, _ = evaluate(model, ldr, criterion, device)
        results[str(group)] = f1

    ctei = compute_cdctei(results)
    print(f'\nCD-CTEI: {ctei:.4f}  (threshold ≥ 0.95)')
    print(f'{"Group":<20} {"Macro F1":>10}')
    print('-' * 32)
    for g, f in sorted(results.items(), key=lambda x: x[1]):
        flag = '⚠' if f < 0.85 else '✓'
        print(f'{g:<20} {f:>10.4f}  {flag}')

    fig, ax = plt.subplots(figsize=(9, 4))
    groups = list(results.keys())
    f1s    = [results[g] for g in groups]
    ax.bar(groups, f1s,
           color=['#2980B9' if f >= 0.85 else '#E74C3C' for f in f1s], alpha=0.85)
    ax.axhline(0.85, color='red', linestyle='--', label='Min acceptable threshold')
    ax.set(ylabel='Macro F1',
           title=f'CD-CTEI={ctei:.3f} — Per-Demographic Stratification Performance')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=15); plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, 'cdctei_equity.png'), dpi=150, bbox_inches='tight')
    plt.show()
    return results, ctei
