# Beyond Binary: Four-Class Risk Stratification from Gastrointestinal Endoscopy Using Asymmetric-Cost CNN-Transformer Learning and Demographic Equity Analysis

**Author:** Azizur Rahman  
**Affiliation:** Indiana Wesleyan University · RadTH Technologies  
**Contact:** azizurusa22@gmail.com  
**Target Venue:** *Computers in Biology and Medicine*

---

## Overview

Gastrointestinal cancers account for over **3.5 million deaths annually**. Early-stage lesions are missed in up to **26% of endoscopic procedures** due to subtle mucosal changes and endoscopist fatigue. Existing AI systems reduce this nuanced clinical decision to a binary output — lesion present or absent — that does not map to any actionable endoscopic classification system and provides no guidance on intervention urgency.

This paper presents the **first CNN-vs-Transformer benchmark for four-class GI lesion risk stratification**, directly aligned with ACG/ESGE clinical practice guidelines, trained under a novel **Asymmetric Endoscopy Loss (AEL)** that penalises missed High-Risk lesions 5× more than Normal misclassification.

---

## Clinical Risk Schema (ACG/ESGE Guideline-Aligned)

| Class | Label | HyperKvasir Source Classes | Clinical Action |
|---|---|---|---|
| 0 | **Normal** | cecum, pylorus, z-line, retroflex-stomach, retroflex-rectum, ileum, bbps-2-3 | Routine surveillance interval (5–10 years) |
| 1 | **Inflammatory** | esophagitis-a, ulcerative-colitis-grade-0-1, ulcerative-colitis-grade-1, hemorrhoids | Medical management + annual follow-up |
| 2 | **Pre-malignant** | barretts, barretts-short-segment, esophagitis-b-d, polyps, ulcerative-colitis-grade-1-2, ulcerative-colitis-grade-2 | Mandatory biopsy + 3–6 month surveillance |
| 3 | **High-Risk** | ulcerative-colitis-grade-2-3, ulcerative-colitis-grade-3, dyed-lifted-polyps, dyed-resection-margins | Immediate resection / oncology referral |

---

## Key Contributions

1. **ACG/ESGE-aligned 4-class schema** — first AI benchmark trained on four clinically actionable risk tiers derived from published society guidelines
2. **Asymmetric Endoscopy Loss (AEL)** — class weights `[1.0, 2.0, 3.0, 5.0]` encoding clinical cost asymmetry; reduces missed High-Risk lesion rate
3. **Lightweight CNN vs. Transformer benchmark** — DenseNet-121, EfficientNet-B0, DeiT-Tiny (<8M params each) compared on identical task under identical training protocol
4. **Cross-dataset generalisation** — zero-shot evaluation on independent Kvasir-v2 cohort
5. **GradCAM explainability** — localises mucosal pit patterns, vascular irregularities, and ulcerated margins per risk tier for both CNN and Transformer models
6. **MC Dropout uncertainty** — 30 stochastic forward passes; flags Pre-malignant/High-Risk for mandatory endoscopist review
7. **AEL ablation study** — AEL vs. Cross-Entropy vs. Focal Loss (γ=2) comparison
8. **Endoscopist workload simulation** — quantifies AI burden reduction with zero missed High-Risk lesions
9. **CD-CTEI fairness** — Cross-Demographic Consistency of Tier-level Equity Index across age (<40, 40–60, >60) and sex subgroups; threshold ≥ 0.95

---

## Architecture Summary

| Model | Type | Pre-training | Parameters | Dropout |
|---|---|---|---|---|
| DenseNet-121 | CNN | ImageNet-1k | 7.0 M | 0.5 |
| EfficientNet-B0 | CNN | ImageNet-1k | 5.3 M | 0.3 |
| DeiT-Tiny | Transformer | ImageNet-1k | 5.9 M | 0.1 |

All models use AdamW (lr=2×10⁻⁴), CosineAnnealingLR, 25 epochs, batch size 32, WeightedRandomSampler for class balance, and CLAHE preprocessing. All three models contain <8M parameters for real-time inference suitability.

---

## Datasets

### Primary Training: HyperKvasir
- **110,079 images** across 23 GI finding classes (labeled image subset used)
- Download: [datasets.simula.no/hyper-kvasir](https://datasets.simula.no/hyper-kvasir/)
- Place at: `data/HyperKvasir/` (zip auto-extracts to `labeled-images/<tract>/<category>/<class>/`)

### External Validation: Kvasir-v2
- **8,000 images** across 8 classes, independently collected
- Download: [datasets.simula.no/kvasir](https://datasets.simula.no/kvasir/)
- Place at: `data/Kvasir-v2/<class-name>/`

---

## Project Structure

```
GastroEndoscopy-Risk-Stratification/
├── GastroEndoscopy_Risk_Stratification.ipynb   # Main research notebook (14 sections)
├── GastroEndoscopy_Paper.docx                  # Full research paper (auto-generated)
├── generate_paper.py                           # Paper generator script
├── README.md
├── requirements.txt
├── checkpoints/
│   ├── densenet121/best.pt
│   ├── efficientnet_b0/best.pt
│   └── deit_tiny/best.pt
└── data/
    ├── HyperKvasir/
    │   └── labeled-images/
    │       ├── lower-gi-tract/
    │       │   ├── anatomical-landmarks/  (cecum, ileum, retroflex-rectum)
    │       │   ├── pathological-findings/ (polyps, ulcerative-colitis-*, hemorrhoids)
    │       │   ├── quality-of-mucosal-views/ (bbps-2-3, ...)
    │       │   └── therapeutic-interventions/ (dyed-lifted-polyps, dyed-resection-margins)
    │       └── upper-gi-tract/
    │           ├── anatomical-landmarks/  (pylorus, retroflex-stomach, z-line)
    │           └── pathological-findings/ (barretts, esophagitis-a, esophagitis-b-d, ...)
    └── Kvasir-v2/
        ├── esophagitis/
        ├── polyps/
        ├── normal-cecum/
        └── ... (8 classes)
```

---

## Notebook Sections

| # | Section | Description |
|---|---|---|
| 1 | Environment Setup | Imports, seeds, device, global constants |
| 2 | Dataset Loading | HyperKvasir loading, 4-class mapping, train/val/test split |
| 3 | Data Augmentation & DataLoader | CLAHE, transforms, WeightedRandomSampler |
| 4 | Model Architecture | DenseNet-121, EfficientNet-B0, DeiT-Tiny builders + sanity check |
| 5 | Asymmetric Endoscopy Loss | AEL definition with clinical weight rationale |
| 6 | Training | `run_training()` for all 3 lightweight models + training curve plots |
| 7 | AEL Ablation Study | AEL vs. Cross-Entropy vs. Focal Loss on DenseNet-121 |
| 8 | Cross-Dataset Evaluation | Kvasir-v2 zero-shot evaluation + confusion matrices |
| 9 | ROC-AUC Curves | Per-class one-vs-rest curves for all trained models |
| 10 | GradCAM Explainability | Saliency maps per risk tier for CNN and Transformer models |
| 11 | Monte Carlo Dropout | Uncertainty quantification + risk-adaptive referral protocol |
| 12 | Workload Simulation | AI burden reduction analysis with threshold sweep |
| 13 | Demographic Equity (CD-CTEI) | Per-subgroup F1 and equity index |
| 14 | Results Summary | Paper tables (Table 2–Table 9) |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download HyperKvasir and Kvasir-v2 (see Datasets section above)

# 3. Launch notebook
jupyter notebook GastroEndoscopy_Risk_Stratification.ipynb

# 4. Run sections in order: 1 → 2 → 3 → 6 (training)
#    Then: 7 (ablation) → 8 (evaluation) → 9–14 (analysis)

# 5. Regenerate paper docx
python generate_paper.py
```

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
opencv-python>=4.8.0
Pillow>=10.0.0
python-docx>=1.0.0
```

---

## Asymmetric Endoscopy Loss (AEL)

```
AEL(ŷ, y) = CrossEntropy(ŷ, y ; w)
where  w = [1.0, 2.0, 3.0, 5.0]
```

| Class | Weight | Rationale |
|---|---|---|
| Normal (0) | 1.0 | Low misclassification cost — unnecessary surveillance only |
| Inflammatory (1) | 2.0 | Missed inflammation delays medical treatment |
| Pre-malignant (2) | 3.0 | Missed biopsy allows unmonitored malignant progression |
| High-Risk (3) | **5.0** | Missed intervention = preventable cancer mortality |

---

## Training Details

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
| Preprocessing | CLAHE (clip=2.0, tile=8×8) |
| Reproducibility | SEED=42, cudnn.deterministic=True |

---

## Expected Output Files

| File | Description |
|---|---|
| `training_curves.png` | Loss and macro F1 curves for all trained models |
| `confusion_matrix_<model>.png` | Count and normalised confusion matrices |
| `roc_curves.png` | Per-class one-vs-rest ROC curves |
| `gradcam_<model>.png` | GradCAM overlays per risk tier |
| `workload_simulation.png` | Pie chart + threshold sensitivity curve |
| `cdctei_equity.png` | Per-subgroup macro F1 bar chart |
| `checkpoints/<model>/best.pt` | Best validation checkpoint |
| `GastroEndoscopy_Paper.docx` | Full research paper |

---

## Citation

If you use this work, please cite:

```bibtex
@article{rahman2026gastrisk,
  title   = {Beyond Binary: Four-Class Risk Stratification from Gastrointestinal
             Endoscopy Using Asymmetric-Cost CNN-Transformer Learning and
             Demographic Equity Analysis},
  author  = {Rahman, Azizur},
  journal = {Computers in Biology and Medicine},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

This project is released for research purposes. Dataset licenses apply per their respective sources:
- HyperKvasir: Creative Commons Attribution 4.0 (CC BY 4.0)
- Kvasir-v2: Creative Commons Attribution 4.0 (CC BY 4.0)
