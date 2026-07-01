"""Regenerate samples_hyperkvasir.png and samples_kvasir-v2.png
with visible class name labels on the left of each row.
"""
import os, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from collections import defaultdict

PROJECT_DIR = '/Users/rahmanazizur/Desktop/GastroEndoscopy-Risk-Stratification'
SEED = 42
random.seed(SEED)

LABEL_NAMES  = ['Normal', 'Inflammatory', 'Pre-malignant', 'High-Risk']
COLORS       = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C']
N_PER_CLASS  = 4

HYPERKVASIR_MAP = {
    'cecum': 0, 'pylorus': 0, 'z-line': 0, 'retroflex-stomach': 0,
    'retroflex-rectum': 0, 'ileum': 0, 'bbps-2-3': 0,
    'esophagitis-a': 1, 'ulcerative-colitis-grade-0-1': 1,
    'ulcerative-colitis-grade-1': 1, 'hemorrhoids': 1,
    'barretts': 2, 'barretts-short-segment': 2, 'esophagitis-b-d': 2,
    'polyps': 2, 'ulcerative-colitis-grade-1-2': 2,
    'ulcerative-colitis-grade-2': 2,
    'ulcerative-colitis-grade-2-3': 3, 'ulcerative-colitis-grade-3': 3,
    'dyed-lifted-polyps': 3, 'dyed-resection-margins': 3,
}
EXCLUDED = {'bbps-0-1', 'impacted-stool', 'out-of-patient', 'short-segment-barretts'}

KVASIR_MAP = {
    'normal-cecum': 0, 'normal-pylorus': 0, 'normal-z-line': 0,
    'esophagitis': 1, 'ulcerative-colitis': 1,
    'polyps': 2, 'barretts': 2,
    'dyed-lifted-polyps': 3, 'dyed-resection-margins': 3,
}


def collect_images(data_dir, label_map, excluded=None):
    excluded = excluded or set()
    class_images = defaultdict(list)
    for root, _, files in os.walk(data_dir):
        folder = os.path.basename(root)
        if folder in excluded:
            continue
        label = label_map.get(folder)
        if label is None:
            continue
        imgs = [os.path.join(root, f) for f in files
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_images[label].extend(imgs)
    return class_images


def make_figure(class_images, title, out_path):
    n_classes = len(LABEL_NAMES)

    # Layout: label column + 4 image columns
    fig = plt.figure(figsize=(N_PER_CLASS * 2.8 + 1.4, n_classes * 2.8))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

    gs = fig.add_gridspec(
        n_classes, N_PER_CLASS + 1,
        width_ratios=[0.38] + [1] * N_PER_CLASS,
        hspace=0.06, wspace=0.04,
        top=0.96, bottom=0.02, left=0.02, right=0.98
    )

    for cls in range(n_classes):
        # ── Label cell ──────────────────────────────────────────────────────
        ax_lbl = fig.add_subplot(gs[cls, 0])
        ax_lbl.set_facecolor(COLORS[cls])
        ax_lbl.text(
            0.5, 0.5, LABEL_NAMES[cls],
            ha='center', va='center',
            fontsize=10, fontweight='bold', color='white',
            rotation=90, transform=ax_lbl.transAxes
        )
        ax_lbl.set_xticks([])
        ax_lbl.set_yticks([])
        for spine in ax_lbl.spines.values():
            spine.set_visible(False)

        # ── Image cells ─────────────────────────────────────────────────────
        imgs   = class_images.get(cls, [])
        sample = random.sample(imgs, min(N_PER_CLASS, len(imgs)))

        for col in range(N_PER_CLASS):
            ax = fig.add_subplot(gs[cls, col + 1])
            if col < len(sample):
                try:
                    ax.imshow(Image.open(sample[col]).convert('RGB'))
                except Exception:
                    ax.set_facecolor('#cccccc')
            else:
                ax.set_facecolor('#cccccc')
            ax.axis('off')

            # thin colored border matching class color
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(COLORS[cls])
                spine.set_linewidth(1.5)

    plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {out_path}')


# ── HyperKvasir ──────────────────────────────────────────────────────────────
hk_dir    = os.path.join(PROJECT_DIR, 'data/HyperKvasir/labeled-images')
hk_images = collect_images(hk_dir, HYPERKVASIR_MAP, EXCLUDED)
make_figure(
    hk_images,
    'HyperKvasir — Sample Images per Risk Class',
    os.path.join(PROJECT_DIR, 'samples_hyperkvasir.png')
)

# ── Kvasir-v2 ────────────────────────────────────────────────────────────────
kv_dir    = os.path.join(PROJECT_DIR, 'data/Kvasir-v2')
kv_images = collect_images(kv_dir, KVASIR_MAP)
make_figure(
    kv_images,
    'Kvasir-v2 — Sample Images per Risk Class',
    os.path.join(PROJECT_DIR, 'samples_kvasir-v2.png')
)

print('Done.')
