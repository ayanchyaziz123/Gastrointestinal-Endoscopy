import os
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

from .config import PROJECT_DIR, NUM_CLASSES, LABEL_NAMES, RISK_LABELS


def visualize_dataset_samples(data_dir, label_map, label_names, n_per_class=4, title='Dataset Samples'):
    from collections import defaultdict
    class_images = defaultdict(list)

    if not os.path.exists(data_dir):
        print(f'[!] Directory not found: {data_dir}'); return

    for root, dirs, files in os.walk(data_dir):
        label = label_map.get(os.path.basename(root))
        if label is None:
            continue
        class_images[label].extend(
            os.path.join(root, f) for f in files
            if f.lower().endswith(('.jpg', '.jpeg', '.png')))

    n_classes = len(label_names)
    colors    = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C']
    fig, axes = plt.subplots(n_classes, n_per_class, figsize=(n_per_class * 3, n_classes * 3))
    axes = np.array(axes).reshape(n_classes, n_per_class)

    for cls in range(n_classes):
        imgs   = class_images.get(cls, [])
        sample = random.sample(imgs, min(n_per_class, len(imgs)))
        for col in range(n_per_class):
            ax = axes[cls, col]
            if col < len(sample):
                try:
                    ax.imshow(Image.open(sample[col]).convert('RGB'))
                except Exception:
                    ax.set_facecolor('#f0f0f0')
            else:
                ax.set_facecolor('#f0f0f0')
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(label_names[cls], fontsize=11, fontweight='bold',
                              color=colors[cls], rotation=90, labelpad=8)

    plt.suptitle(f'{title} — Sample Images per Risk Class', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, f'samples_{title.replace(" ","_").lower()}.png'),
                dpi=100, bbox_inches='tight')
    plt.show()


def plot_class_distribution(data_dir, label_map, label_names, title='Class Distribution'):
    from collections import defaultdict
    counts = defaultdict(int)

    if not os.path.exists(data_dir):
        print(f'[!] Directory not found: {data_dir}'); return

    for root, dirs, files in os.walk(data_dir):
        label = label_map.get(os.path.basename(root))
        if label is None:
            continue
        counts[label] += sum(1 for f in files
                             if f.lower().endswith(('.jpg', '.jpeg', '.png')))

    total = sum(counts.values())
    if total == 0:
        print(f'[!] No images found in {data_dir}'); return

    colors = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C']
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    bars = axes[0].bar([label_names[i] for i in range(len(label_names))],
                       [counts[i] for i in range(len(label_names))],
                       color=colors, alpha=0.85, edgecolor='white')
    for bar, i in zip(bars, range(len(label_names))):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f'{counts[i]}\n({100*counts[i]/total:.1f}%)',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[0].set(ylabel='Image Count', title=title)
    axes[0].grid(axis='y', alpha=0.3)
    axes[1].pie([counts[i] for i in range(len(label_names))],
                labels=label_names, colors=colors, autopct='%1.1f%%',
                startangle=90, pctdistance=0.8)
    axes[1].set_title(f'{title} — Pie')
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, f'dist_{title.replace(" ","_").lower()}.png'),
                dpi=100, bbox_inches='tight')
    plt.show()


def plot_imbalance_handling():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C']
    labels = ['Normal', 'Inflammatory', 'Pre-malignant', 'High-Risk']
    x, w   = np.arange(len(labels)), 0.35

    raw    = [5252, 645, 1836, 2152]
    capped = [3000, 645, 1836, 2152]
    axes[0].bar(x - w/2, raw,    w, label='Raw',   color=colors, alpha=0.35, edgecolor='black')
    axes[0].bar(x + w/2, capped, w, label='Capped', color=colors, alpha=0.9,  edgecolor='black')
    for i, val in enumerate(capped):
        axes[0].text(x[i] + w/2, val + 30, str(val), ha='center', fontsize=8, fontweight='bold')
    axes[0].axhline(3000, color='red', linestyle='--', linewidth=1.2, label='max_normal=3000')
    axes[0].set(xticks=x, xticklabels=labels, ylabel='Image Count', title='① Normal Cap')
    axes[0].tick_params(axis='x', rotation=12); axes[0].legend(fontsize=8); axes[0].grid(axis='y', alpha=0.3)

    counts       = np.array([3000, 645, 1836, 2152], dtype=float)
    proportional = counts / counts.sum()
    balanced     = (1.0 / counts) / (1.0 / counts).sum()
    axes[1].bar(x - w/2, proportional, w, label='Without sampler', color=colors, alpha=0.35, edgecolor='black')
    axes[1].bar(x + w/2, balanced,     w, label='WeightedSampler',  color=colors, alpha=0.9,  edgecolor='black')
    axes[1].axhline(0.25, color='red', linestyle='--', linewidth=1.2, label='Ideal 25%')
    axes[1].set(xticks=x, xticklabels=labels, ylabel='Fraction per Batch', title='② WeightedRandomSampler')
    axes[1].tick_params(axis='x', rotation=12); axes[1].legend(fontsize=7); axes[1].grid(axis='y', alpha=0.3)

    ael  = [1.0, 2.0, 3.0, 5.0]
    bars = axes[2].bar(labels, ael, color=colors, alpha=0.9, edgecolor='black')
    for bar, val in zip(bars, ael):
        axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.06,
                     f'{val}×', ha='center', fontsize=13, fontweight='bold')
    axes[2].set(ylabel='Loss Penalty', ylim=(0, 6.2), title='③ AEL Weights')
    axes[2].tick_params(axis='x', rotation=12); axes[2].grid(axis='y', alpha=0.3)

    plt.suptitle('Three-Stage Imbalance Strategy: Cap → Sampler → AEL',
                 fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, 'imbalance_handling.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_training_curves(histories: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ['#2980B9', '#E74C3C', '#2ECC71', '#F39C12']
    for (name, h), color in zip(histories.items(), colors):
        if not h:
            continue
        axes[0].plot(h['train_loss'], '--', color=color, alpha=0.6, label=f'{name} train')
        axes[0].plot(h['val_loss'],   '-',  color=color, label=f'{name} val')
        axes[1].plot(h['train_f1'],   '--', color=color, alpha=0.6)
        axes[1].plot(h['val_f1'],     '-',  color=color, label=name)
    for ax, title, ylabel in zip(axes, ['AEL Loss Curves', 'Macro F1 Curves'], ['Loss', 'Macro F1']):
        ax.set(xlabel='Epoch', ylabel=ylabel, title=title)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.suptitle('Training History — CNN vs Transformer (HyperKvasir)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_curves(results_dict: dict):
    colors = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C']
    styles = ['-', '--', ':', '-.']
    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(5 * NUM_CLASSES, 5))

    for c, (cls_name, risk, ax) in enumerate(zip(LABEL_NAMES, RISK_LABELS, axes)):
        for (model_name, results), ls in zip(results_dict.items(), styles):
            trues_bin = [1 if t == c else 0 for t in results['trues']]
            probs_cls = np.array(results['probs'])[:, c]
            fpr, tpr, _ = roc_curve(trues_bin, probs_cls)
            ax.plot(fpr, tpr, linestyle=ls, linewidth=2,
                    label=f'{model_name} (AUC={auc(fpr, tpr):.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax.set(xlabel='FPR', ylabel='TPR', title=f'{cls_name}\n({risk})')
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        if c == 3:
            ax.text(0.3, 0.05, 'HIGH PRIORITY\nImmediate Intervention',
                    fontsize=8, color='red', alpha=0.8,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Per-Class ROC Curves — GI Lesion Risk Stratification (One-vs-Rest)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, 'roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.show()


def print_auc_table(results_dict: dict):
    print(f'\n{"Model":<22}', end='')
    for c in LABEL_NAMES:
        print(f'{c:>16}', end='')
    print(f'{"Macro AUC":>12}')
    print('-' * 90)
    for model_name, results in results_dict.items():
        trues_bin = label_binarize(results['trues'], classes=list(range(NUM_CLASSES)))
        aucs = []
        print(f'{model_name:<22}', end='')
        for c in range(NUM_CLASSES):
            a = roc_auc_score(trues_bin[:, c], np.array(results['probs'])[:, c])
            aucs.append(a)
            print(f'{a:>16.4f}', end='')
        print(f'{np.mean(aucs):>12.4f}')
