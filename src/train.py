import copy
import json
import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score
from tqdm import tqdm

from .config import DEVICE, AEL_WEIGHTS, CKPT_DIR
from .loss import AsymmetricEndoscopyLoss


def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss, preds_all, trues_all = 0.0, [], []
    for imgs, labels in tqdm(loader, desc='  batches', leave=True):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        preds_all.extend(out.detach().argmax(1).cpu().numpy())
        trues_all.extend(labels.cpu().numpy())
    scheduler.step()
    return (total_loss / len(loader),
            f1_score(trues_all, preds_all, average='macro', zero_division=0))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds_all, trues_all = 0.0, [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        total_loss += criterion(out, labels).item()
        preds_all.extend(out.argmax(1).cpu().numpy())
        trues_all.extend(labels.cpu().numpy())
    return (f1_score(trues_all, preds_all, average='macro', zero_division=0),
            total_loss / len(loader), preds_all, trues_all)


def _save_history(history, hist_path):
    tmp = hist_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(history, f)
    os.replace(tmp, hist_path)


def _save_checkpoint(model_state, optimizer_state, scheduler_state,
                     epoch, best_f1, history, ckpt_dir):
    """Atomic save of full training state to enable resume after crash."""
    resume_path = os.path.join(ckpt_dir, 'resume.pt')
    tmp = resume_path + '.tmp'
    torch.save({
        'epoch':           epoch,
        'best_f1':         best_f1,
        'model_state':     model_state,
        'optimizer_state': optimizer_state,
        'scheduler_state': scheduler_state,
    }, tmp)
    os.replace(tmp, resume_path)


def _save_best(model_state, ckpt_path):
    tmp = ckpt_path + '.tmp'
    torch.save(model_state, tmp)
    os.replace(tmp, ckpt_path)


def run_training(model, train_ldr, val_ldr, model_name,
                 epochs=25, lr=2e-4, ckpt_dir=CKPT_DIR):
    ckpt_path   = os.path.join(ckpt_dir, model_name, 'best.pt')
    hist_path   = os.path.join(ckpt_dir, model_name, 'history.json')
    resume_path = os.path.join(ckpt_dir, model_name, 'resume.pt')
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    model     = model.to(DEVICE)
    criterion = AsymmetricEndoscopyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    start_epoch = 1
    best_f1     = 0.0
    history     = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}

    if os.path.exists(resume_path):
        state     = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        completed = state['epoch']
        if completed >= epochs:
            print(f'[DONE] {model_name} already completed {completed}/{epochs} epochs.')
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
            with open(hist_path) as f:
                history = json.load(f)
            return model, history
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        scheduler.load_state_dict(state['scheduler_state'])
        best_f1     = state['best_f1']
        start_epoch = completed + 1
        with open(hist_path) as f:
            history = json.load(f)
        print(f'[RESUME] {model_name} — epoch {start_epoch}/{epochs}  '
              f'(best val F1 so far: {best_f1:.4f})')

    elif os.path.exists(ckpt_path):
        completed_epochs = (len(json.load(open(hist_path)).get('train_loss', []))
                            if os.path.exists(hist_path) else 0)
        if completed_epochs >= epochs:
            print(f'[DONE] {model_name} — loading finished checkpoint.')
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
            with open(hist_path) as f:
                history = json.load(f)
            return model, history
        print('[WARN] Checkpoint found but no resume state. Re-training from scratch.')

    else:
        print(f'\n{"="*60}')
        print(f'  Training : {model_name}')
        print(f'  Device   : {DEVICE}  |  Epochs: {epochs}  |  LR: {lr}')
        print(f'  Loss     : AsymmetricEndoscopyLoss  weights={AEL_WEIGHTS}')
        print(f'{"="*60}')

    for epoch in range(start_epoch, epochs + 1):
        print(f'\nEpoch {epoch}/{epochs}')
        tl, tf       = train_epoch(model, train_ldr, optimizer, scheduler, criterion, DEVICE)
        vf, vl, _, _ = evaluate(model, val_ldr, criterion, DEVICE)

        history['train_loss'].append(tl)
        history['val_loss'].append(vl)
        history['train_f1'].append(tf)
        history['val_f1'].append(vf)

        _save_checkpoint(copy.deepcopy(model.state_dict()),
                         optimizer.state_dict(), scheduler.state_dict(),
                         epoch, best_f1, history, os.path.dirname(ckpt_path))
        _save_history(history, hist_path)

        marker = ''
        if vf > best_f1:
            best_f1 = vf
            _save_best(copy.deepcopy(model.state_dict()), ckpt_path)
            marker = '  *** NEW BEST ***'

        print(f'  loss={tl:.4f}  train_F1={tf:.4f}  val_loss={vl:.4f}  val_F1={vf:.4f}{marker}')

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    print(f'\n  Done. Best val F1 = {best_f1:.4f}  →  {ckpt_path}')
    return model, history
