import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
def stratify_labels_for_binary_with_rule(df, rule_col, target_col):
    keys = (df[rule_col].astype(str) + "_" + df[target_col].astype(int).astype(str)).values
    _, inv = np.unique(keys, return_inverse=True)
    return inv
