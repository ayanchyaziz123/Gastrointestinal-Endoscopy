import os

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from .config import PROJECT_DIR, DEVICE, IMG_SIZE, LABEL_NAMES, RISK_LABELS
from .dataset import val_transforms, DATASET_MEAN, DATASET_STD


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.grads = None
        self.acts  = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'acts', o.detach()))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'grads', go[0].detach()))

    def generate(self, img_tensor, class_idx=None):
        self.model.eval()
        img = img_tensor.unsqueeze(0).to(DEVICE)
        out = self.model(img)
        if class_idx is None:
            class_idx = out.argmax().item()
        self.model.zero_grad()
        out[0, class_idx].backward()

        if self.grads.dim() == 4:
            # CNN path: (B, C, H, W)
            weights = self.grads.mean(dim=[2, 3], keepdim=True)
            cam     = F.relu((weights * self.acts).sum(1, keepdim=True))
            cam     = cam.squeeze().cpu().numpy()
        else:
            # Transformer path: (B, N, C) — reshape token sequence to 2-D spatial grid
            weights = self.grads.mean(dim=1)
            cam     = F.relu((weights.unsqueeze(1) * self.acts).sum(-1))
            cam     = cam.squeeze().cpu().numpy()
            n       = int(cam.shape[0] ** 0.5)
            cam     = cam[:n * n].reshape(n, n)

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        return cam, class_idx


def get_target_layer(model, model_name):
    if model_name == 'densenet121':
        return model.features.denseblock4.denselayer16.conv2
    elif model_name == 'efficientnet_b0':
        return model.conv_head
    elif model_name == 'deit_tiny':
        return model.blocks[-1].norm1
    raise ValueError(f'Unknown model: {model_name}')


def visualize_gradcam(model, model_name, image_paths, true_labels=None):
    inv = transforms.Normalize(
        mean=[-m / s for m, s in zip(DATASET_MEAN, DATASET_STD)],
        std=[1 / s for s in DATASET_STD])

    target_layer = get_target_layer(model, model_name)
    gcam         = GradCAM(model, target_layer)

    n      = len(image_paths)
    colors = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C']
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 9))
    axes = np.array(axes).reshape(2, n)

    for i, path in enumerate(image_paths):
        img_t     = val_transforms(Image.open(path).convert('RGB'))
        cam, pred = gcam.generate(img_t)
        img_np    = np.array(inv(img_t).permute(1, 2, 0).clamp(0, 1))
        heatmap   = cv2.cvtColor(
            cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET),
            cv2.COLOR_BGR2RGB) / 255.0
        overlay   = np.clip(0.6 * img_np + 0.4 * heatmap, 0, 1)

        gt = LABEL_NAMES[true_labels[i]] if true_labels else ''
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f'GT: {gt}', fontsize=10)
        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f'{LABEL_NAMES[pred]}\n{RISK_LABELS[pred]}',
                              fontsize=9, color=colors[pred])
        for ax in [axes[0, i], axes[1, i]]:
            ax.axis('off')

    plt.suptitle(f'GradCAM — {model_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(PROJECT_DIR, f'gradcam_{model_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.show()
