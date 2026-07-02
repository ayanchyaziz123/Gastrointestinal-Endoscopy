import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DEVICE, LABEL_NAMES, RISK_LABELS


def mc_predict(model, img_tensor, n_samples=30):
    """T=30 stochastic forward passes with dropout active for uncertainty estimation."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    img   = img_tensor.unsqueeze(0).to(DEVICE)
    probs = []
    with torch.no_grad():
        for _ in range(n_samples):
            probs.append(F.softmax(model(img), dim=1).cpu().numpy())

    model.eval()  # restore full eval mode

    probs      = np.stack(probs).squeeze(1)   # (n_samples, NUM_CLASSES)
    mean_probs = probs.mean(axis=0)
    pred       = int(mean_probs.argmax())
    confidence = float(mean_probs.max())
    entropy    = float(-np.sum(mean_probs * np.log(mean_probs + 1e-8)))
    # Pre-malignant and High-Risk always flagged regardless of confidence
    flag       = confidence < 0.75 or pred >= 2

    return {
        'prediction':  pred,
        'label':       LABEL_NAMES[pred],
        'risk':        RISK_LABELS[pred],
        'confidence':  confidence,
        'uncertainty': entropy,
        'flag':        flag,
        'mean_probs':  mean_probs,
    }
