import arff
import pandas as pd
import numpy as np
import os

DATA_DIR = "/home/claude/data/NASADefectDataset-master/CleanedData/MDP/D''"
FILES = ['CM1.arff','KC1.arff','JM1.arff','PC1.arff','PC3.arff','PC4.arff','KC3.arff','MW1.arff']

COMMON = ['BRANCH_COUNT', 'CYCLOMATIC_COMPLEXITY', 'DESIGN_COMPLEXITY', 'ESSENTIAL_COMPLEXITY',
          'HALSTEAD_CONTENT', 'HALSTEAD_DIFFICULTY', 'HALSTEAD_EFFORT', 'HALSTEAD_ERROR_EST',
          'HALSTEAD_LENGTH', 'HALSTEAD_LEVEL', 'HALSTEAD_PROG_TIME', 'HALSTEAD_VOLUME',
          'LOC_BLANK', 'LOC_CODE_AND_COMMENT', 'LOC_COMMENTS', 'LOC_EXECUTABLE', 'LOC_TOTAL',
          'NUM_OPERANDS', 'NUM_OPERATORS', 'NUM_UNIQUE_OPERANDS', 'NUM_UNIQUE_OPERATORS']

def load_one(fname):
    path = os.path.join(DATA_DIR, fname)
    with open(path) as fh:
        d = arff.load(fh)
    cols = [a[0] for a in d['attributes']]
    df = pd.DataFrame(d['data'], columns=cols)
    # normalize label column name
    if 'Defective' in df.columns:
        label_col = 'Defective'
    elif 'label' in df.columns:
        label_col = 'label'
    else:
        raise ValueError(f"No label column found in {fname}: {cols}")
    df = df.rename(columns={label_col: 'Defective'})
    df['project'] = fname.replace('.arff', '')
    keep = COMMON + ['Defective', 'project']
    return df[keep]

frames = [load_one(f) for f in FILES]
raw = pd.concat(frames, ignore_index=True)
raw['Defective_bin'] = (raw['Defective'] == 'Y').astype(int)

print("=== RAW COMBINED DATASET ===")
print("Total rows:", len(raw))
print("Rows per project:")
print(raw.groupby('project').size())
print()
print("Defect rate per project:")
print(raw.groupby('project')['Defective_bin'].mean().round(4) * 100)
print()
print("Overall defect rate: {:.2f}%".format(raw['Defective_bin'].mean()*100))

raw.to_csv("/home/claude/data/combined_raw.csv", index=False)
print("\nSaved combined_raw.csv, shape:", raw.shape)
