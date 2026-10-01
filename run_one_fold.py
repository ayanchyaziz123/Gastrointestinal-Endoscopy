import os
os.environ['MPLBACKEND'] = 'Agg'
import nbformat
from nbclient import NotebookClient

notebook_path = "GastroEndoscopy_Risk_Stratification_new.ipynb"

# Load the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# 1. Fix the data loading cell: look for the line with 'df = load_hyperkvasir'
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        source = ''.join(cell.get('source', []))
        if 'df = load_hyperkvasir' in source:
            lines = cell['source']
            # Find the index of the line that contains 'df = load_hyperkvasir'
            for i, line in enumerate(lines):
                if 'df = load_hyperkvasir' in line:
                    # We will replace the lines after the df assignment to ensure we have at least one sample per class.
                    # We'll remove the existing lines after the df assignment (up to the print statement) and replace with:
                    #   # Ensure we have at least 5 samples per class for splitting (if available)
                    #   df = df.groupby('label').apply(lambda x: x.sample(n=min(5, len(x)), random_state=SEED)).reset_index(drop=True)
                    #   print(f'After balancing: {len(df)} samples')
                    # Then we keep the train_test_split lines (without stratify) and the print statement.
                    # We'll remove from i+1 until we reach the line that starts with 'if len(df) > 0:' (which is the train_test_split line).
                    # Actually, let's just replace the next 4 lines (the if len(df) > 1000 block and the two train_test_split lines and the print) with our new block.
                    # But note: the structure might have changed. We'll do a more robust replacement by looking for the line that starts with 'if len(df) > 0:' and replacing from i+1 to that line.
                    # Find the index of the line that starts with 'if len(df) > 0:'
                    j = i+1
                    while j < len(lines) and not lines[j].strip().startswith('if len(df) > 0:'):
                        j += 1
                    # Now we want to replace lines[i+1:j] with our new lines.
                    new_lines = [
                        '# Ensure we have at least 5 samples per class for splitting (if available)\\n',
                        'df = df.groupby(\\'label\\').apply(lambda x: x.sample(n=min(5, len(x)), random_state=SEED)).reset_index(drop=True)\\n',
                        'print(f\\'After balancing: {len(df)} samples\\')\\n'
                    ]
                    # Replace the slice
                    lines[i+1:j] = new_lines
                    break
            cell['source'] = lines
            break

# 2. Fix the 5-fold CV cell (index 54) to run only one fold and one model
target_index = 54  # the 5-fold CV cell
if target_index < len(nb['cells']):
    cell = nb['cells'][target_index]
    lines = cell['source']
    # We want to change the loop over MODELS to only run 'densenet121'
    # And the loop over folds to only run fold_idx=0
    # We'll do this by changing the lists and then breaking after the first iteration.
    # Instead, we can change the loop to:
    #   for model_name in ['densenet121']:
    #       ... and then inside the fold loop, break after the first fold.
    # Let's find the line with "MODELS       = ['densenet121']"
    for i, line in enumerate(lines):
        if line.strip() == "MODELS       = ['densenet121']   # Only run 5-fold for densenet121, keep efficientnet_b0 and deit_tiny as is":
            # We'll change it to just run densenet121 (it already is) and then we'll break after the first fold.
            # We'll leave the MODELS line as is (it's already only densenet121).
            pass
        # We'll also change the loop over folds to break after the first fold.
        # We'll look for the line: "for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(df, df['label'])):"
        if line.strip() == "for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(df, df['label'])):":
            # We'll change it to:
            #   for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(df, df['label'])):
            #       if fold_idx > 0:
            #           break
            # We'll insert a break condition after the line.
            # We'll insert two lines after this line.
            lines.insert(i+1, '        if fold_idx > 0:')
            lines.insert(i+2, '            break')
            break
    # Also, we want to set MAX_EPOCHS to 1 (already done in previous fixes, but let's ensure)
    for i, line in enumerate(lines):
        if line.strip() == "N_FOLDS      = 5":
            # Set the next line to MAX_EPOCHS = 1
            if i+1 < len(lines):
                lines[i+1] = "MAX_EPOCHS = 1   # Reduced for quick run\\n"
            break
    cell['source'] = lines
else:
    print(f"Warning: 5-fold CV cell index {target_index} is out of bounds")

# Save the modified notebook to a temporary file
temp_notebook_path = "GastroEndoscopy_Risk_Stratification_one_fold.ipynb"
with open(temp_notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

# Create a notebook client with reasonable timeouts
client = NotebookClient(nb, timeout=30, startup_timeout=10)  # 30 seconds per cell, 10 seconds startup

try:
    client.execute()
except Exception as e:
    print(f"Error during execution: {e}")
    # Optionally, save the notebook as is
    with open(temp_notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    raise

# Save the executed notebook (optional)
with open(temp_notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook execution completed for one fold.")