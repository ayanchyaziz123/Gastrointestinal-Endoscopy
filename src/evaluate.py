import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, f1_score, recall_score, confusion_matrix,
)
from tqdm import tqdm

from .config import PROJECT_DIR, DEVICE, NUM_CLASSES, LABEL_NAMES, RISK_ACTIONS


@torch.no_grad()
def evaluate_detailed(model, loader, device, dataset_name=''):
    model.eval()
    preds_all, trues_all, probs_all = [], [], []
    for imgs, labels in tqdm(loader, desc=f'Evaluating {dataset_name}', leave=True):
        out = model(imgs.to(device))
        probs_all.extend(F.softmax(out, dim=1).cpu().numpy())
        preds_all.extend(out.argmax(1).cpu().numpy())
        trues_all.extend(labels.numpy())
    macro_f1         = f1_score(trues_all, preds_all, average='macro', zero_division=0)
    per_class        = f1_score(trues_all, preds_all, average=None,    zero_division=0)
    high_risk_recall = recall_score(
        [1 if t == 3 else 0 for t in trues_all],
        [1 if p == 3 else 0 for p in preds_all], zero_division=0)
    print(f'\n{"="*60}')
    print(f'Dataset: {dataset_name}  |  Macro F1: {macro_f1:.4f}  |  HR Recall: {high_risk_recall:.4f}')
    print(classification_report(trues_all, preds_all, target_names=LABEL_NAMES, zero_division=0))
    return {'macro_f1': macro_f1, 'per_class': per_class,
            'high_risk_recall': high_risk_recall,
            'preds': preds_all, 'trues': trues_all,
            'probs': np.array(probs_all)}


def plot_confusion_matrix(results, model_name):
    cm  = confusion_matrix(results['trues'], results['preds'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    short = ['Normal', 'Inflam.', 'Pre-mal.', 'High-Risk']
    for ax, data, fmt, title in zip(
            [ax1, ax2],
            [cm, cm.astype(float) / cm.sum(axis=1, keepdims=True)],
            ['d', '.1%'],
            ['Count', 'Normalized']):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                    xticklabels=short, yticklabels=short)
        ax.set(xlabel='Predicted', ylabel='True',
               title=f'{model_name} Confusion Matrix ({title})')
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, f'confusion_matrix_{model_name}.png'),
                dpi=150, bbox_inches='tight')
    plt.show()


def print_paper_tables(results_dict: dict, ext_results: dict = None):
    print('\nTable 2. Per-Class F1 Score — All Models')
    print('=' * 95)
    print(f'{"Class":<22}', end='')
    for name in results_dict:
        print(f'{name:>18}', end='')
    print('  Clinical Action')
    print('-' * 95)
    for i, (label, action) in enumerate(zip(LABEL_NAMES, RISK_ACTIONS)):
        print(f'{label:<22}', end='')
        for res in results_dict.values():
            print(f'{res["per_class"][i]:>18.4f}', end='')
        print(f'  {action}')
    print('-' * 95)
    print(f'{"Macro F1":<22}', end='')
    for res in results_dict.values():
        print(f'{res["macro_f1"]:>18.4f}', end='')
    print()
    print(f'{"High-Risk Recall":<22}', end='')
    for res in results_dict.values():
        print(f'{res["high_risk_recall"]:>18.4f}', end='')
    print()

    if ext_results:
        print('\nTable 3. Cross-Dataset Generalization (Kvasir-v2)')
        print('=' * 70)
        print(f'{"Dataset":<20}', end='')
        for name in results_dict:
            print(f'{name:>18}', end='')
        print()
        print('-' * 70)
        for site, res_by_model in ext_results.items():
            print(f'{site:<20}', end='')
            for name in results_dict:
                print(f'{res_by_model.get(name, {}).get("macro_f1", 0):>18.4f}', end='')
            print()


def calibrate_thresholds(model, val_ldr, device):
    """Find per-class probability threshold that maximises macro F1 on validation set."""
    model.eval()
    probs_all, trues_all = [], []
    with torch.no_grad():
        for imgs, labels in val_ldr:
            out = model(imgs.to(device))
            probs_all.extend(F.softmax(out, dim=1).cpu().numpy())
            trues_all.extend(labels.numpy())
    probs_all = np.array(probs_all)
    trues_all = np.array(trues_all)

    best_thresholds = np.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        best_f1, best_t = 0.0, 0.5
        for t in np.arange(0.1, 0.9, 0.02):
            preds = np.where(probs_all[:, c] >= t, c,
                             np.argmax(np.where(np.arange(NUM_CLASSES) == c,
                                                -np.inf, probs_all), axis=1))
            f1 = f1_score((trues_all == c).astype(int),
                          (preds == c).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thresholds[c] = best_t
        print(f'  Class {c} ({LABEL_NAMES[c]:<18}): threshold={best_t:.2f}  val F1={best_f1:.4f}')
    return best_thresholds


def predict_with_thresholds(model, loader, device, thresholds):
    """Predict using per-class calibrated thresholds instead of argmax."""
    model.eval()
    probs_all, trues_all = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            out = model(imgs.to(device))
            probs_all.extend(F.softmax(out, dim=1).cpu().numpy())
            trues_all.extend(labels.numpy())
    probs_all = np.array(probs_all)
    margins   = probs_all - thresholds[np.newaxis, :]
    preds     = np.argmax(margins, axis=1)
    macro_f1  = f1_score(trues_all, preds, average='macro', zero_division=0)
    print(f'\nWith calibrated thresholds — Macro F1: {macro_f1:.4f}')
    print(classification_report(trues_all, preds, target_names=LABEL_NAMES, zero_division=0))
    return preds, macro_f1
