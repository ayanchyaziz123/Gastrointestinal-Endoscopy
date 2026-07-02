"""Generate framework diagram PNG for the paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(18, 6.5))
ax.set_xlim(0, 18)
ax.set_ylim(2.8, 9)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Colour palette ────────────────────────────────────────────────────────────
C_INPUT   = '#E8F4FD'
C_PRE     = '#FFF3CD'
C_MODEL   = '#D4EDDA'
C_LOSS    = '#F8D7DA'
C_OUT     = '#D1ECF1'
C_RISK    = ['#28a745', '#fd7e14', '#ffc107', '#dc3545']
C_ARROW   = '#495057'
C_BORDER  = '#343a40'

def box(ax, x, y, w, h, label, sublabel='', color='#FFFFFF', fontsize=10,
        subfontsize=8, radius=0.25, bold=False):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle=f'round,pad=0.05,rounding_size={radius}',
                          linewidth=1.5, edgecolor=C_BORDER,
                          facecolor=color, zorder=3)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ty = y + h/2 + (0.15 if sublabel else 0)
    ax.text(x + w/2, ty, label, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, zorder=4, color='#212529')
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.22, sublabel, ha='center', va='center',
                fontsize=subfontsize, color='#495057', zorder=4)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=C_ARROW,
                                lw=1.8, mutation_scale=16), zorder=5)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(9, 8.6, 'GI Endoscopy Risk Stratification Framework',
        ha='center', va='center', fontsize=15, fontweight='bold', color='#212529')

# ── Stage 1: Input ────────────────────────────────────────────────────────────
box(ax, 0.3, 5.5, 2.2, 1.6, 'Endoscopy', 'Image Input',
    color=C_INPUT, fontsize=10, bold=True)

# ── Stage 2: CLAHE ────────────────────────────────────────────────────────────
box(ax, 3.0, 5.5, 2.2, 1.6, 'CLAHE', 'Preprocessing',
    color=C_PRE, fontsize=10, bold=True)
arrow(ax, 2.5, 6.3, 3.0, 6.3)

# label
ax.text(2.75, 6.55, '224×224', ha='center', fontsize=7.5, color='#6c757d')

# ── Stage 3: Three architectures ──────────────────────────────────────────────
archs = [
    ('DenseNet-121', '7.0 M params'),
    ('EfficientNet-B0', '5.3 M params'),
    ('DeiT-Tiny', '5.9 M params'),
]
arch_y = [7.0, 5.8, 4.6]
for (name, params), y in zip(archs, arch_y):
    box(ax, 5.9, y, 2.6, 0.9, name, params,
        color=C_MODEL, fontsize=9, subfontsize=7.5, bold=True)
    arrow(ax, 5.2, 6.3, 5.9, y + 0.45)

ax.text(5.55, 6.3, 'Split', ha='center', fontsize=8, color='#6c757d',
        style='italic')

# Bracket / brace label
ax.text(5.2, 6.8, 'Lightweight\nArchitectures', ha='center', fontsize=8,
        color='#343a40', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#f8f9fa',
                  edgecolor='#adb5bd', linewidth=0.8))

# ── Stage 4: AEL Loss ─────────────────────────────────────────────────────────
box(ax, 9.2, 5.5, 2.5, 1.6,
    'Asymmetric\nEndoscopy Loss',
    'w = [1.0, 3.5, 3.0, 5.0]',
    color=C_LOSS, fontsize=9.5, subfontsize=7.5, bold=True)

for y in arch_y:
    arrow(ax, 8.5, y + 0.45, 9.2, 6.3)

ax.text(8.85, 6.3, 'Train', ha='center', fontsize=8,
        color='#6c757d', style='italic')

# ── Stage 5: MC Dropout ───────────────────────────────────────────────────────
box(ax, 12.1, 5.5, 2.5, 1.6,
    'MC Dropout',
    'T = 30 passes  τ = 0.75',
    color=C_OUT, fontsize=9.5, subfontsize=7.5, bold=True)
arrow(ax, 11.7, 6.3, 12.1, 6.3)

# ── Stage 6: Four risk outputs ────────────────────────────────────────────────
risk_labels = [
    ('Normal', 'Routine surveillance', '#28a745'),
    ('Inflammatory', 'Medical management', '#fd7e14'),
    ('Pre-malignant', 'Biopsy required', '#ffc107'),
    ('High-Risk', 'Immediate intervention', '#dc3545'),
]
risk_y = [7.5, 6.5, 5.5, 4.5]
for (label, action, col), y in zip(risk_labels, risk_y):
    box(ax, 15.2, y, 2.55, 0.75, label, action,
        color=col + '33', fontsize=8.5, subfontsize=7, bold=True,
        radius=0.15)
    # colour dot
    circ = plt.Circle((15.05, y + 0.375), 0.12, color=col, zorder=6)
    ax.add_patch(circ)
    arrow(ax, 14.6, 6.3, 15.2, y + 0.375)

ax.text(14.9, 6.3, 'Classify', ha='center', fontsize=8,
        color='#6c757d', style='italic')

# ── GradCAM note ──────────────────────────────────────────────────────────────
box(ax, 5.9, 3.2, 5.8, 0.85,
    'GradCAM Explainability  →  Clinically interpretable saliency maps per risk tier',
    color='#F3E5F5', fontsize=8.5, radius=0.15)

arrow(ax, 8.8, 5.5, 8.8, 4.05)

# ── Auto-clear / escalate note ────────────────────────────────────────────────
box(ax, 12.1, 3.2, 5.65, 0.85,
    'Auto-clear 43.9% low-confidence cases  |  Zero missed High-Risk lesions',
    color='#E8F5E9', fontsize=8.5, radius=0.15)

arrow(ax, 14.9, 5.5, 14.9, 4.05)

# ── Dataset note ──────────────────────────────────────────────────────────────
box(ax, 0.3, 3.2, 4.8, 0.85,
    'HyperKvasir (10,662 imgs)  +  Kvasir-v2 zero-shot validation',
    color='#E3F2FD', fontsize=8.5, radius=0.15)

arrow(ax, 1.4, 5.5, 1.4, 4.05)

# ── Workflow label ────────────────────────────────────────────────────────────
for xi, label in zip([1.4, 4.1, 7.2, 10.45, 13.35],
                     ['① Input', '② Preprocess', '③ Encode', '④ Train (AEL)', '⑤ Infer']):
    ax.text(xi, 5.25, label, ha='center', fontsize=8,
            color='#6c757d', style='italic')

# ── Bottom legend ─────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_INPUT,  edgecolor=C_BORDER, label='Input'),
    mpatches.Patch(facecolor=C_PRE,    edgecolor=C_BORDER, label='Preprocessing'),
    mpatches.Patch(facecolor=C_MODEL,  edgecolor=C_BORDER, label='Model architectures'),
    mpatches.Patch(facecolor=C_LOSS,   edgecolor=C_BORDER, label='Loss (AEL)'),
    mpatches.Patch(facecolor=C_OUT,    edgecolor=C_BORDER, label='Uncertainty (MC Dropout)'),
    mpatches.Patch(facecolor='#dc354533', edgecolor='#dc3545', label='Risk output'),
]
ax.legend(handles=legend_items, loc='lower center', ncol=6,
          fontsize=8, framealpha=0.9,
          bbox_to_anchor=(0.5, 0.0), frameon=True)

plt.tight_layout(pad=0.5)
out = '/Users/rahmanazizur/Desktop/GastroEndoscopy-Risk-Stratification/framework.png'
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f'Saved: {out}')
plt.close()
