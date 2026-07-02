import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, recall_score
from tqdm import tqdm

from .config import DEVICE
from .models import build_densenet121
from .loss import AsymmetricEndoscopyLoss, FocalLoss


def run_ablation(train_ldr, val_ldr, test_ldr, epochs=10):
    """Compare AEL vs Cross-Entropy vs Focal Loss on DenseNet-121."""
    loss_fns = {
        'Cross-Entropy': nn.CrossEntropyLoss(),
        'Focal (γ=2)':   FocalLoss(gamma=2.0),
        'AEL (Ours)':    AsymmetricEndoscopyLoss(),
    }
    results = {}
    for name, loss_fn in loss_fns.items():
        print(f'\n>>> Ablation: {name}')
        model = build_densenet121().to(DEVICE)
        opt   = AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        sch   = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            bar = tqdm(train_ldr, desc=f'{name} Ep{epoch}/{epochs}', unit='batch', leave=False)
            for imgs, labels in bar:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                opt.zero_grad()
                loss = loss_fn(model(imgs), labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                running_loss += loss.item()
                bar.set_postfix(loss=f'{loss.item():.4f}')
            sch.step()
            print(f'  Ep {epoch}/{epochs} — avg loss: {running_loss/len(train_ldr):.4f}')

        model.eval()
        preds_all, trues_all = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(test_ldr, desc=f'  {name} eval', leave=False):
                preds_all.extend(model(imgs.to(DEVICE)).argmax(1).cpu().numpy())
                trues_all.extend(labels.numpy())

        macro_f1         = f1_score(trues_all, preds_all, average='macro', zero_division=0)
        high_risk_recall = recall_score(
            [1 if t == 3 else 0 for t in trues_all],
            [1 if p == 3 else 0 for p in preds_all], zero_division=0)
        results[name] = {'macro_f1': macro_f1, 'high_risk_recall': high_risk_recall}
        print(f'  Done — Macro F1={macro_f1:.4f}  HR Recall={high_risk_recall:.4f}')

    print('\nAblation Summary:')
    print(f'{"Loss":<20} {"Macro F1":>10} {"HR Recall":>12}')
    print('-' * 44)
    for k, v in results.items():
        print(f'{k:<20} {v["macro_f1"]:>10.4f} {v["high_risk_recall"]:>12.4f}')
    return results
