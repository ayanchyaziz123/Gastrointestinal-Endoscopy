"""
Generate the MBEC graphical abstract.
Target: 32.93 mm × 37.63 mm at 300 dpi  →  389 × 444 pixels
Output: graphical_abstract.png

Run from project root:
    python generate_graphical_abstract.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── canvas: 32.93 mm × 37.63 mm at 300 dpi ───────────────────────────────────
DPI    = 300
W_MM, H_MM = 32.93, 37.63
W_IN   = W_MM / 25.4
H_IN   = H_MM / 25.4

fig, ax = plt.subplots(figsize=(W_IN, H_IN), dpi=DPI)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── colour palette ─────────────────────────────────────────────────────────────
C_NORMAL   = '#27AE60'
C_INFLAM   = '#F39C12'
C_PREMAL   = '#E67E22'
C_HIGH     = '#C0392B'
C_ARROW    = '#2C3E50'
C_BOX_BG   = '#ECF0F1'
C_TITLE    = '#2C3E50'

FS_TITLE   = 3.2
FS_BODY    = 2.4
FS_SMALL   = 2.0

# ── title ──────────────────────────────────────────────────────────────────────
ax.text(0.5, 0.96, 'GI Endoscopy Risk Stratification',
        ha='center', va='top', fontsize=FS_TITLE, fontweight='bold',
        color=C_TITLE)
ax.text(0.5, 0.91, 'CNN–Transformer + Asymmetric Endoscopy Loss',
        ha='center', va='top', fontsize=FS_SMALL, color='#555555')

# ── thin divider ───────────────────────────────────────────────────────────────
ax.axhline(0.88, xmin=0.05, xmax=0.95, color='#BDC3C7', linewidth=0.4)

# ── pipeline boxes ────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, label, sublabel='', color='#ECF0F1', tc='#2C3E50'):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle='round,pad=0.005',
                           facecolor=color, edgecolor='#95A5A6', linewidth=0.3)
    ax.add_patch(rect)
    ax.text(x, y + (0.012 if sublabel else 0), label,
            ha='center', va='center', fontsize=FS_BODY, fontweight='bold', color=tc)
    if sublabel:
        ax.text(x, y - 0.016, sublabel,
                ha='center', va='center', fontsize=FS_SMALL-0.3, color=tc)

def arrow(ax, x1, y, x2):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=C_ARROW,
                                lw=0.5, mutation_scale=4))

# Row 1: Input → Preprocessing → Models
ROW1_Y = 0.76
box(ax, 0.12, ROW1_Y, 0.18, 0.07, 'HyperKvasir', '7,633 images', C_BOX_BG)
arrow(ax, 0.215, ROW1_Y, 0.295)
box(ax, 0.36, ROW1_Y, 0.18, 0.07, 'CLAHE', 'Augmentation', C_BOX_BG)
arrow(ax, 0.455, ROW1_Y, 0.535)
box(ax, 0.70, ROW1_Y, 0.26, 0.07, '3 Architectures',
    'DenseNet · EffNet · DeiT', '#D6EAF8')

# Row 2: AEL loss
ROW2_Y = 0.62
ax.text(0.5, ROW2_Y + 0.04, 'Asymmetric Endoscopy Loss  AEL',
        ha='center', va='center', fontsize=FS_BODY, fontweight='bold', color='#154360')
ax.text(0.5, ROW2_Y + 0.015,
        r'$\mathbf{w}=[1.0,\ 3.5,\ 3.0,\ 5.0]$  |  High-Risk penalty $5\times$',
        ha='center', va='center', fontsize=FS_SMALL, color='#154360')

# Row 3: 4 risk classes
ROW3_Y = 0.46
ax.text(0.5, ROW3_Y + 0.075, 'Four-Class Risk Schema  (ACG/ESGE)',
        ha='center', va='center', fontsize=FS_BODY, fontweight='bold', color=C_TITLE)

for i, (lbl, sub, col) in enumerate([
    ('Normal',       'Surveillance',  C_NORMAL),
    ('Inflammatory', 'Medical Mgmt',  C_INFLAM),
    ('Pre-malignant','Biopsy',        C_PREMAL),
    ('High-Risk',    'Resection',     C_HIGH),
]):
    x = 0.12 + i * 0.255
    rect = FancyBboxPatch((x - 0.10, ROW3_Y - 0.04), 0.20, 0.07,
                           boxstyle='round,pad=0.005',
                           facecolor=col, edgecolor='none', linewidth=0)
    ax.add_patch(rect)
    ax.text(x, ROW3_Y + 0.01, lbl,  ha='center', va='center',
            fontsize=FS_SMALL, fontweight='bold', color='white')
    ax.text(x, ROW3_Y - 0.02, sub, ha='center', va='center',
            fontsize=FS_SMALL - 0.4, color='white')

# Row 4: Key results
ax.axhline(0.39, xmin=0.05, xmax=0.95, color='#BDC3C7', linewidth=0.4)
ROW4_Y = 0.30

results = [
    ('Macro F1', '0.84', '#2980B9'),
    ('Zero\nMissed HR', '✓', '#27AE60'),
    ('Workload\nReduction', '44.9%', '#8E44AD'),
    ('ECE', '0.08–0.09', '#16A085'),
]
for i, (metric, val, col) in enumerate(results):
    x = 0.12 + i * 0.255
    ax.text(x, ROW4_Y + 0.035, val, ha='center', va='center',
            fontsize=FS_TITLE, fontweight='bold', color=col)
    ax.text(x, ROW4_Y - 0.005, metric, ha='center', va='center',
            fontsize=FS_SMALL - 0.2, color='#555555')

# Row 5: MC Dropout + GradCAM
ax.axhline(0.22, xmin=0.05, xmax=0.95, color='#BDC3C7', linewidth=0.4)
ax.text(0.5, 0.17,
        'MC Dropout (T=30, τ=0.75)  +  GradCAM Explainability',
        ha='center', va='center', fontsize=FS_SMALL, color='#555555')
ax.text(0.5, 0.11,
        'Cross-dataset zero-shot: Kvasir-v2  |  Macro F1 = 0.78',
        ha='center', va='center', fontsize=FS_SMALL, color='#555555')

ax.axhline(0.07, xmin=0.05, xmax=0.95, color='#BDC3C7', linewidth=0.3)
ax.text(0.5, 0.04,
        'HyperKvasir  ·  7,633 images  ·  DenseNet-121 / EfficientNet-B0 / DeiT-Tiny',
        ha='center', va='center', fontsize=FS_SMALL - 0.3, color='#888888')

# ── save ───────────────────────────────────────────────────────────────────────
out = os.path.join(PROJECT_DIR, 'graphical_abstract.png')
plt.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
print(f'Saved → {out}')
print(f'Size at 300 dpi: {W_MM:.2f} mm × {H_MM:.2f} mm')
