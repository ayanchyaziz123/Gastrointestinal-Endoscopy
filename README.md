# Beyond Binary: Four-Class Risk Stratification from Gastrointestinal Endoscopy Using Asymmetric-Cost Lightweight CNN–Transformer Learning and Demographic Equity Analysis

**Author:** Azizur Rahman  
**Affiliation:** Indiana Wesleyan University · RadTH Technologies  
**Contact:** azizurusa22@gmail.com  
**Target Venue:** *Medical & Biological Engineering & Computing* (Springer, journal 11517)

---

## Overview

Gastrointestinal cancers cause over **3.5 million deaths annually**. Early-stage lesions are missed in up to **26% of endoscopic procedures** due to subtle mucosal changes and endoscopist fatigue. Existing AI systems reduce the clinical decision to a binary output — lesion/no lesion — that does not map to any actionable classification system and gives no guidance on intervention urgency.

This work presents the **first CNN-vs-Transformer benchmark for four-class GI lesion risk stratification** directly aligned with ACG/ESGE clinical practice guidelines. Training is driven by a novel **Asymmetric Endoscopy Loss (AEL)** that penalises missed High-Risk lesions 5× more than Normal misclassification, and inference is equipped with MC Dropout uncertainty quantification and GradCAM explainability.

---

## Clinical Risk Schema (ACG/ESGE Aligned)

| Class | Label | HyperKvasir Source Classes | Clinical Action |
|---|---|---|---|
| 0 | **Normal** | cecum, pylorus, z-line, retroflex-stomach, retroflex-rectum, ileum, bbps-2-3 | Routine surveillance (5–10 yr) |
| 1 | **Inflammatory** | esophagitis-a, ulcerative-colitis-grade-0-1, ulcerative-colitis-grade-1, hemorrhoids | Medical management + annual endoscopy |
| 2 | **Pre-malignant** | barretts, barretts-short-segment, esophagitis-b-d, polyps, ulcerative-colitis-grade-1-2, ulcerative-colitis-grade-2 | Mandatory biopsy + 3–6 month surveillance |
| 3 | **High-Risk** | ulcerative-colitis-grade-2-3, ulcerative-colitis-grade-3, dyed-lifted-polyps, dyed-resection-margins | Immediate resection / oncology referral |

---

## Key Contributions

1. **ACG/ESGE-aligned 4-class schema** — first AI benchmark on four clinically actionable risk tiers from published society guidelines
2. **Asymmetric Endoscopy Loss (AEL)** — weights `[1.0, 3.5, 3.0, 5.0]` encoding clinical cost asymmetry; zero missed High-Risk lesions on test set across all architectures
3. **Lightweight CNN vs. Transformer benchmark** — DenseNet-121, EfficientNet-B0, DeiT-Tiny (all < 8 M params) on identical protocol; 14 ms single-image inference on Apple M2
4. **Cross-dataset generalisation** — zero-shot evaluation on the independent Kvasir-v2 cohort; macro F1 = 0.76
5. **GradCAM explainability** — per-risk-tier saliency maps localising mucosal pit patterns and vascular irregularities for both CNN and Transformer models
6. **MC Dropout uncertainty** — T=30 stochastic passes; τ=0.75 confidence threshold flags Pre-malignant/High-Risk for mandatory endoscopist review
7. **AEL ablation** — AEL vs. Cross-Entropy vs. Focal Loss (γ=2) on DenseNet-121
8. **Endoscopist workload simulation** — quantifies AI burden reduction with zero missed High-Risk cases
9. **CD-CTEI fairness framework** — proof-of-concept equity audit across demographic subgroups; scaffold for studies with real patient annotations

---

## Architecture Summary

| Model | Type | Pre-training | Parameters | Dropout |
|---|---|---|---|---|
| DenseNet-121 | CNN | ImageNet-1k | 7.0 M | 0.5 |
| EfficientNet-B0 | CNN | ImageNet-1k | 5.3 M | 0.3 |
| DeiT-Tiny | Transformer | ImageNet-1k | 5.9 M | 0.1 |

All models: AdamW (lr=2×10⁻⁴, wd=1×10⁻⁴), CosineAnnealingLR, 25 epochs, batch 32, WeightedRandomSampler, CLAHE preprocessing.

---

## Asymmetric Endoscopy Loss (AEL)

```
AEL(ŷ, y) = CrossEntropy(ŷ, y ; w)
  w = [1.0, 3.5, 3.0, 5.0]
```

| Class | Weight | Rationale |
|---|---|---|
| Normal (0) | 1.0 | Misclassification cost is low |
| Inflammatory (1) | **3.5** | Raised from 2.0 to compensate for small class size (645 images) |
| Pre-malignant (2) | 3.0 | Missed biopsy allows unmonitored progression |
| High-Risk (3) | **5.0** | Missed intervention = preventable cancer mortality |

---

## Datasets

### Primary: HyperKvasir
- **7,633 usable images** (Normal: 3,000 capped, Inflammatory: 645, Pre-malignant: 1,836, High-Risk: 2,152)
- Download: [datasets.simula.no/hyper-kvasir](https://datasets.simula.no/hyper-kvasir/)
- Place at: `data/HyperKvasir/` (extracts to `labeled-images/<tract>/<category>/<class>/`)

### External Validation: Kvasir-v2
- **8,000 images** across 8 classes, independently collected
- Download: [datasets.simula.no/kvasir](https://datasets.simula.no/kvasir/)
- Place at: `data/Kvasir-v2/<class-name>/`

**Kvasir-v2 → 4-tier mapping:**

| Kvasir-v2 folder | Risk tier |
|---|---|
| normal-cecum, normal-pylorus, normal-z-line | Normal (0) |
| esophagitis | Inflammatory (1) |
| ulcerative-colitis, polyps, barretts | Pre-malignant (2) |
| dyed-lifted-polyps, dyed-resection-margins | High-Risk (3) |

---

## Project Structure

```
GastroEndoscopy-Risk-Stratification/
│
├── main.py                                  # Full pipeline runner (train → eval → explainability)
├── ael_sensitivity.py                       # AEL High-Risk weight sensitivity sweep (w = 3–7)
│
├── src/                                     # Modular source package
│   ├── config.py                            # Seeds, DEVICE, hyperparams, label names
│   ├── dataset.py                           # HYPERKVASIR_MAP, CLAHETransform, EndoscopyDataset, loaders
│   ├── models.py                            # build_densenet121 / efficientnet_b0 / deit_tiny
│   ├── loss.py                              # AsymmetricEndoscopyLoss, FocalLoss
│   ├── train.py                             # train_epoch, evaluate, run_training (crash-safe checkpointing)
│   ├── evaluate.py                          # evaluate_detailed, confusion matrix, threshold calibration
│   ├── uncertainty.py                       # mc_predict (MC Dropout, T=30)
│   ├── explainability.py                    # GradCAM, get_target_layer, visualize_gradcam
│   ├── workload.py                          # endoscopist_simulation
│   ├── equity.py                            # compute_cdctei, equity_analysis
│   ├── ablation.py                          # run_ablation (AEL vs CE vs Focal)
│   └── visualize.py                         # All plot functions
│
├── GastroEndoscopy_Risk_Stratification.ipynb  # Interactive notebook (14 sections, unchanged)
├── paper.tex                                # LaTeX manuscript (MBEC submission)
├── requirements.txt
│
├── checkpoints/                             # Saved per-epoch resume states + best.pt
│   ├── densenet121/
│   ├── efficientnet_b0/
│   └── deit_tiny/
│
└── data/
    ├── HyperKvasir/labeled-images/
    └── Kvasir-v2/
```

---

## Quick Start

```bash
# 1. Create and activate virtual environment
python3 -m venv --system-site-packages venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download datasets (see links above) and place under data/

# 4a. Run the full pipeline
python main.py

# 4b. Or use the interactive notebook
jupyter notebook GastroEndoscopy_Risk_Stratification.ipynb

# 5. Run AEL weight sensitivity sweep (optional, ~5 hours on M2)
python ael_sensitivity.py
```

Training is **crash-safe**: re-running any training cell or `main.py` resumes from the last completed epoch via `checkpoints/<model>/resume.pt`.

---

## Training Hyperparameters

| Hyperparameter | Value |
|---|---|
| Optimiser | AdamW |
| Learning rate | 2×10⁻⁴ |
| Weight decay | 1×10⁻⁴ |
| LR schedule | CosineAnnealingLR (η_min=1×10⁻⁶) |
| Epochs | 25 |
| Batch size | 32 |
| Gradient clipping | 1.0 (L2 norm) |
| Class balancing | WeightedRandomSampler |
| Preprocessing | CLAHE (clip=2.0, tile=8×8) + ImageNet normalisation |
| Seed | 42 |

---

## Output Files

| File | Description |
|---|---|
| `training_curves.png` | Loss and macro F1 curves for all models |
| `confusion_matrix_<model>.png` | Count and normalised confusion matrices |
| `roc_curves.png` | Per-class one-vs-rest ROC-AUC curves |
| `gradcam_<model>.png` | GradCAM overlays — one image per risk tier |
| `workload_simulation.png` | Triage pie chart + threshold sensitivity sweep |
| `imbalance_handling.png` | Cap → Sampler → AEL three-stage strategy |
| `ael_sensitivity_results.csv` | HR weight sweep results (w = 3.0 – 7.0) |
| `checkpoints/<model>/best.pt` | Best validation checkpoint |

---

## Notebook Sections

| # | Section |
|---|---|
| 1 | Environment setup — imports, seeds, device, constants |
| 2 | Dataset loading — HyperKvasir 4-class mapping, train/val/test split |
| 3 | Data augmentation — CLAHE, transforms, WeightedRandomSampler |
| 4 | Model architecture — DenseNet-121, EfficientNet-B0, DeiT-Tiny |
| 5 | Asymmetric Endoscopy Loss (AEL) |
| 6 | Training — all three models, crash-safe checkpointing |
| 7 | AEL ablation — AEL vs. Cross-Entropy vs. Focal Loss |
| 8 | Cross-dataset evaluation — Kvasir-v2 zero-shot + confusion matrices |
| 9 | ROC-AUC curves |
| 10 | GradCAM explainability |
| 11 | Monte Carlo Dropout uncertainty |
| 12 | Endoscopist workload simulation |
| 13 | Demographic equity (CD-CTEI) |
| 14 | Results summary and paper tables |

---

## Citation

```bibtex
@article{rahman2026gastrisk,
  title   = {Beyond Binary: Four-Class Risk Stratification from Gastrointestinal
             Endoscopy Using Asymmetric-Cost Lightweight CNN--Transformer Learning
             and Demographic Equity Analysis},
  author  = {Rahman, Azizur},
  journal = {Medical \& Biological Engineering \& Computing},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

Released for research purposes. Dataset licenses apply per their respective sources:
- HyperKvasir: Creative Commons Attribution 4.0 (CC BY 4.0)
- Kvasir-v2: Creative Commons Attribution 4.0 (CC BY 4.0)
