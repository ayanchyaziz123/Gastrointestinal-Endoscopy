import json

notebook_path = "GastroEndoscopy_Risk_Stratification_new.ipynb"

# Load the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Fix the data loading cell: look for the line with 'df = load_hyperkvasir'
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        source = ''.join(cell.get('source', []))
        if 'df = load_hyperkvasir' in source:
            lines = cell['source']
            # Find the index of the line that contains 'df = load_hyperkvasir'
            for i, line in enumerate(lines):
                if 'df = load_hyperkvasir' in line:
                    # We will replace the lines from i+1 to the end of the cell?
                    # Instead, we will replace the line after the df assignment with our new lines.
                    # We want to keep the df assignment line, then add two lines:
                    #   # Ensure we have at least 2 samples per class for splitting
                    #   df = df.groupby('label').apply(lambda x: x.sample(n=min(20, len(x)), random_state=SEED)).reset_index(drop=True)
                    # But note: we also want to keep the print statement that follows?
                    # Let's look at the original structure after the df assignment:
                    #   if len(df) > 1000:
                    #       df = df.sample(n=1000, random_state=SEED)
                    #   if len(df) > 0:
                    #       train_df, temp  = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=SEED)
                    #       val_df, test_df = train_test_split(temp, test_size=0.5, stratify=temp['label'], random_state=SEED)
                    #       print(f'\nSplit: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}')
                    # We want to remove the if len(df) > 1000 block and replace it with our balancing step.
                    # We'll remove the lines from i+1 until we reach the line that starts with 'if len(df) > 0:' (which is the train_test_split line).
                    # Actually, we want to keep the train_test_split lines but without the if len(df) > 0 guard?
                    # Let's just replace the two lines after the df assignment (which are the if len(df) > 1000 block) with our balancing step and then keep the rest.
                    # We'll do:
                    #   line i+1: '# Ensure we have at least 2 samples per class for splitting\n'
                    #   line i+2: 'df = df.groupby(\'label\').apply(lambda x: x.sample(n=min(20, len(x)), random_state=SEED)).reset_index(drop=True)\n'
                    # Then we leave the rest of the cell as is (starting from the original line i+3?).
                    # But note: the original line i+1 is 'if len(df) > 1000:\n' and i+2 is '    df = df.sample(n=1000, random_state=SEED)\n'
                    # We want to remove these two lines and insert our two lines.
                    if i+1 < len(lines):
                        del lines[i+1]
                    if i+1 < len(lines):  # after first deletion, the next line shifts
                        del lines[i+1]
                    # Insert our two lines
                    lines.insert(i+1, '# Ensure we have at least 2 samples per class for splitting\\n')
                    lines.insert(i+2, 'df = df.groupby(\\'label\\').apply(lambda x: x.sample(n=min(20, len(x)), random_state=SEED)).reset_index(drop=True)\\n')
                    # Update the cell's source
                    cell['source'] = lines
                    break
            break

# 2. Fix the 5-fold CV cell (by index 54, but let's also try by id for safety)
target_id = "9c27f8ac-e58b-46f8-81cd-16d0e2cfd352"
target_index = None
for i, cell in enumerate(nb['cells']):
    if cell.get('id') == target_id:
        target_index = i
        break

if target_index is None:
    # Fallback to index 54 if we can't find by id
    target_index = 54

if target_index is not None and target_index < len(nb['cells']):
    cell = nb['cells'][target_index]
    lines = cell['source']
    # We want to fix lines starting from the line that contains "N_FOLDS      = 5"
    for i, line in enumerate(lines):
        if line.strip() == "N_FOLDS      = 5":
            # We have found the N_FOLDS line at index i.
            # We want to set:
            #   i+1: MAX_EPOCHS = 1   # Reduced for quick run
            #   i+2: RESULTS_DIR  = os.path.join(PROJECT_DIR, 'results')
            #   i+3: os.makedirs(RESULTS_DIR, exist_ok=True)
            # But note: the current lines at i+1 and beyond are messed up.
            # We'll replace the lines from i+1 up to (but not including) the line that starts with "LABEL_NAMES_SHORT".
            # Find the index of the line that starts with "LABEL_NAMES_SHORT"
            j = i+1
            while j < len(lines) and not lines[j].strip().startswith("LABEL_NAMES_SHORT"):
                j += 1
            # Now we want to replace lines[i+1:j] with our three lines, each ending with newline.
            new_lines = [
                "MAX_EPOCHS = 1   # Reduced for quick run\\n",
                "RESULTS_DIR  = os.path.join(PROJECT_DIR, 'results')\\n",
                "os.makedirs(RESULTS_DIR, exist_ok=True)\\n"
            ]
            # Replace the slice
            lines[i+1:j] = new_lines
            break
    cell['source'] = lines
else:
    print(f"Warning: Could not find 5-fold CV cell (index {target_index} is out of bounds)")

# Save the notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Fixed {notebook_path}")