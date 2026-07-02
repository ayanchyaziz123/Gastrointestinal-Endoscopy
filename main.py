"""
GastroEndoscopy Risk Stratification — full pipeline runner.

Usage:
    python main.py
"""

import json
import os

from torch.utils.data import DataLoader

from src.config import PROJECT_DIR, DEVICE, LABEL_NAMES
from src.dataset import (
    load_hyperkvasir, load_kvasir_v2, split_dataset, make_loaders,
    EndoscopyDataset, val_transforms, HYPERKVASIR_MAP, KVASIR_V2_MAP,
)
from src.models import build_densenet121, build_efficientnet_b0, build_deit_tiny
from src.train import run_training
from src.evaluate import (
    evaluate_detailed, plot_confusion_matrix, print_paper_tables,
    calibrate_thresholds, predict_with_thresholds,
)
from src.ablation import run_ablation
from src.visualize import (
    visualize_dataset_samples, plot_imbalance_handling,
    plot_training_curves, plot_roc_curves, print_auc_table,
)
from src.explainability import visualize_gradcam
from src.workload import endoscopist_simulation

DATA_ROOT   = os.path.join(PROJECT_DIR, 'data', 'HyperKvasir')
KVASIR_ROOT = os.path.join(PROJECT_DIR, 'data', 'Kvasir-v2')


def main():
    # ── 1. Dataset ───────────────────────────────────────────────────────────
    df = load_hyperkvasir(DATA_ROOT)
    if len(df) == 0:
        raise RuntimeError(f'No images found at {DATA_ROOT}. Check the data path.')
    train_df, val_df, test_df = split_dataset(df)
    train_ldr, val_ldr, test_ldr = make_loaders(train_df, val_df, test_df)

    # ── 2. Visualize dataset ─────────────────────────────────────────────────
    visualize_dataset_samples(os.path.join(DATA_ROOT, 'labeled-images'),
                              HYPERKVASIR_MAP, LABEL_NAMES)
    plot_imbalance_handling()

    # ── 3. Train ─────────────────────────────────────────────────────────────
    trained_models, histories = {}, {}
    for name, builder in [('densenet121',     build_densenet121),
                          ('efficientnet_b0', build_efficientnet_b0),
                          ('deit_tiny',       build_deit_tiny)]:
        trained_models[name], histories[name] = run_training(
            builder(), train_ldr, val_ldr, name)
        plot_training_curves(histories)

    # ── 4. Evaluate — internal test set ─────────────────────────────────────
    results = {}
    for name, model in trained_models.items():
        results[name] = evaluate_detailed(model, test_ldr, DEVICE,
                                          dataset_name=f'{name} (HyperKvasir test)')
        plot_confusion_matrix(results[name], name)

    # ── 5. Cross-dataset evaluation (Kvasir-v2) ───────────────────────────────
    ext_results = {}
    kv2_df = load_kvasir_v2(KVASIR_ROOT)
    if len(kv2_df) > 0:
        kv2_ldr = DataLoader(EndoscopyDataset(kv2_df, val_transforms),
                             32, shuffle=False, num_workers=0)
        ext_results['Kvasir-v2'] = {}
        for name, model in trained_models.items():
            ext_results['Kvasir-v2'][name] = evaluate_detailed(
                model, kv2_ldr, DEVICE, dataset_name=f'{name} (Kvasir-v2)')

    # ── 6. ROC curves ─────────────────────────────────────────────────────────
    plot_roc_curves(results)
    print_auc_table(results)

    # ── 7. Ablation ───────────────────────────────────────────────────────────
    ablation_results = run_ablation(train_ldr, val_ldr, test_ldr, epochs=10)
    ablation_path = os.path.join(PROJECT_DIR, 'checkpoints', 'ablation_results.json')
    os.makedirs(os.path.dirname(ablation_path), exist_ok=True)
    with open(ablation_path, 'w') as f:
        json.dump(ablation_results, f, indent=2)

    # ── 8. GradCAM ────────────────────────────────────────────────────────────
    sample_paths, sample_labels = [], []
    for label in range(4):
        rows = test_df[test_df['label'] == label]
        if len(rows) > 0:
            row = rows.iloc[0]
            sample_paths.append(row['image_path'])
            sample_labels.append(label)
    for name, model in trained_models.items():
        visualize_gradcam(model, name, sample_paths, true_labels=sample_labels)

    # ── 9. Workload simulation ─────────────────────────────────────────────────
    best_name  = max(results, key=lambda n: results[n]['macro_f1'])
    best_model = trained_models[best_name]
    print(f'Workload simulation using: {best_name}')
    endoscopist_simulation(best_model, test_df, confidence_threshold=0.75)

    # ── 10. Calibrated thresholds ──────────────────────────────────────────────
    thresholds = calibrate_thresholds(best_model, val_ldr, DEVICE)
    predict_with_thresholds(best_model, test_ldr, DEVICE, thresholds)

    # ── 11. Paper tables ───────────────────────────────────────────────────────
    print_paper_tables(results, ext_results)


if __name__ == '__main__':
    main()
