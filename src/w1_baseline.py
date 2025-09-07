import os, argparse, json, yaml
import pandas as pd, numpy as np
from src.logging_utils import configure_logger
from src.utils import build_stratified_folds, tune_threshold, metrics_report
from src.nli_scorer import compute_nli_scores

def ensure_parent(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f: cfg = yaml.safe_load(f)
    logger = configure_logger("./logs", name="w1_baseline")
    paths = cfg["paths"]; text_col = cfg["text_col"]; rule_col = cfg["rule_col"]; target_col = cfg["target_col"]
    n_splits = int(cfg.get("n_splits", 5)); seed = int(cfg.get("seed", 42))
    proc_train = paths["processed_train_csv"]
    if not os.path.exists(proc_train): logger.error(f"processed_train_csv not found: {proc_train}"); raise SystemExit(1)
    df = pd.read_csv(proc_train)
    for col in [text_col, rule_col, target_col]:
        if col not in df.columns: raise ValueError(f"Missing column {col} in processed_train_csv")
    nli_path = paths["nli_scores_train"]
    if os.path.exists(nli_path):
        nli = pd.read_csv(nli_path)["nli_entail_prob"].values; logger.info(f"Loaded NLI: {nli_path}")
    else:
        ncfg = cfg["nli"]
        nli = compute_nli_scores(df, text_col, rule_col, ncfg["model"], ncfg["variants"], ncfg.get("batch_size", 8), logger)
        ensure_parent(nli_path); pd.DataFrame({"nli_entail_prob": nli}).to_csv(nli_path, index=False, encoding="utf-8-sig")
        logger.info(f"Saved NLI -> {nli_path}")
    df["nli_prob"] = nli
    folds = build_stratified_folds(df, rule_col, target_col, n_splits, seed)
    oof = np.zeros(len(df), dtype=float); thr_list = []
    for i, (tr, va) in enumerate(folds):
        y_true = df.iloc[va][target_col].astype(int).values
        y_prob = df.iloc[va]["nli_prob"].values
        thr = tune_threshold(y_true, y_prob); thr_list.append(thr); oof[va] = y_prob
        logger.info(f"[FOLD {i}] best_thr={thr:.3f}")
    global_thr = tune_threshold(df[target_col].astype(int).values, oof)
    rep = metrics_report(df[target_col].astype(int).values, oof, global_thr); rep["global_threshold"] = float(global_thr)
    ensure_parent(paths["oof_csv"]); ensure_parent(paths["thresholds_json"]); ensure_parent(paths["baseline_report"])
    pd.DataFrame({"nli_prob": oof}).to_csv(paths["oof_csv"], index=False, encoding="utf-8-sig")
    import json
    with open(paths["thresholds_json"], "w", encoding="utf-8") as f: json.dump({"fold_thresholds": thr_list, "global_threshold": global_thr}, f, ensure_ascii=False, indent=2)
    with open(paths["baseline_report"], "w", encoding="utf-8") as f: json.dump(rep, f, ensure_ascii=False, indent=2)
    logger.info(f"[W1] Baseline report: {rep}")
    logger.info("W1 baseline finished.")
if __name__ == "__main__": main()
