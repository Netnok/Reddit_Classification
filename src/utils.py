from __future__ import annotations
import numpy as np, pandas as pd
from typing import Dict, List, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_recall_fscore_support

def build_stratified_folds(df: pd.DataFrame, rule_col: str, target_col: str, n_splits: int, seed: int):
    keys = (df[rule_col].astype(str) + "_" + df[target_col].astype(int).astype(str)).values
    _, inv = np.unique(keys, return_inverse=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(inv)), inv))

def tune_threshold(y_true, y_prob) -> float:
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 181):
        y_hat = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, y_hat, average="macro", zero_division=0)
        if f1 > best_f1: best_f1, best_thr = float(f1), float(thr)
    return best_thr

def metrics_report(y_true, y_prob, thr: float):
    y_hat = (y_prob >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_hat, average="macro", zero_division=0)
    return {"precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f1)}
