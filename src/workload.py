import os

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from .config import PROJECT_DIR
from .dataset import val_transforms
from .uncertainty import mc_predict


def endoscopist_simulation(model, test_df, confidence_threshold=0.75):
    """
    Simulate AI-assisted triage: count auto-cleared vs endoscopist-reviewed cases.
    All High-Risk and Pre-malignant predictions are always flagged for review.
    """
    auto_cleared, flagged = 0, 0
    missed_high_risk      = 0
    flag_breakdown        = {'low_confidence': 0, 'pre_malignant': 0, 'high_risk': 0}

    mc_results = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc='Simulating'):
        img = val_transforms(Image.open(row['image_path']).convert('RGB'))
        mc_results.append((mc_predict(model, img), row['label']))

    for res, true_label in mc_results:
        auto = res['confidence'] >= confidence_threshold and res['prediction'] < 2
        if not auto:
            flagged += 1
            if res['confidence'] < confidence_threshold: flag_breakdown['low_confidence'] += 1
            if res['prediction'] == 2:                   flag_breakdown['pre_malignant']  += 1
            if res['prediction'] == 3:                   flag_breakdown['high_risk']       += 1
        else:
            auto_cleared += 1
            if true_label == 3:
                missed_high_risk += 1

    total = len(test_df)
    print(f'\nEndoscopist Workload Simulation  (threshold={confidence_threshold})')
    print(f'  Total patients:           {total}')
    print(f'  AI auto-cleared:          {auto_cleared} ({100*auto_cleared/total:.1f}%)')
    print(f'  Flagged for endoscopist:  {flagged} ({100*flagged/total:.1f}%)')
    print(f'  Workload reduction:       {100*auto_cleared/total:.1f}%')
    print(f'  Missed High-Risk:         {missed_high_risk}  (target: 0)')
    print(f'  Flag breakdown:           {flag_breakdown}')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    ax1.pie([auto_cleared, flagged],
            labels=[f'AI auto-cleared\n({100*auto_cleared/total:.1f}%)',
                    f'Endoscopist review\n({100*flagged/total:.1f}%)'],
            colors=['#2ECC71', '#E74C3C'], autopct='%1.1f%%', startangle=90)
    ax1.set_title('Endoscopy Screening Workflow')

    thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    reductions = [100 * sum(1 for res, _ in mc_results
                            if res['confidence'] >= t and res['prediction'] < 2) / total
                  for t in thresholds]
    ax2.plot(thresholds, reductions, 'b-o', linewidth=2)
    ax2.axvline(confidence_threshold, color='red', linestyle='--',
                label=f'threshold={confidence_threshold}')
    ax2.set(xlabel='Confidence Threshold', ylabel='Workload Reduction (%)',
            title='Threshold vs. Burden Reduction')
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, 'workload_simulation.png'), dpi=150, bbox_inches='tight')
    plt.show()
    return auto_cleared, flagged, missed_high_risk
