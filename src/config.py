import os
import random
import warnings

import numpy as np
import torch

warnings.filterwarnings('ignore')

# Repo root (parent of this src/ directory)
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR    = os.path.join(PROJECT_DIR, 'checkpoints')

SEED        = 42
NUM_CLASSES = 4
IMG_SIZE    = 224
BATCH_SIZE  = 32

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE  = (8, 8)

# Inflammatory raised to 3.5 to compensate for small class size (645 images)
AEL_WEIGHTS = [1.0, 3.5, 3.0, 5.0]   # Normal, Inflammatory, Pre-malignant, High-Risk

DROPOUT_RATES = {
    'densenet121':     0.5,
    'efficientnet_b0': 0.3,
    'deit_tiny':       0.1,
}

LABEL_NAMES  = ['Normal', 'Inflammatory', 'Pre_malignant', 'High_Risk']
RISK_LABELS  = ['Routine Surveillance', 'Medical Management',
                'Biopsy + Surveillance', 'Immediate Intervention']
RISK_ACTIONS = [
    'Continue standard screening interval',
    'Medical treatment + annual endoscopy',
    'Biopsy required + 3-6 month follow-up',
    'Resection or oncology referral',
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
