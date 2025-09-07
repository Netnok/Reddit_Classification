import argparse, json, os, re
import pandas as pd
from typing import Dict, List, Any
import yaml
from logging_utils import configure_logger

def soft_normalize(text: str) -> str:
    if not isinstance(text, str): text = str(text)
    t = re.sub(r'(?:([A-Za-z])\.)+(?=[A-Za-z])', lambda m: m.group(0).replace('.', ''), text)
    if re.search(r'[A-Za-z]\.[A-Za-z]', t): t = t.replace('.', '')
    t = t.translate(str.maketrans({'0':'o','1':'i','3':'e','4':'a','5':'s','7':'t'}))
    for zw in ["\u200b","\u200c","\u200d","\ufeff"]: t = t.replace(zw,"")
    return re.sub(r'\s+', ' ', t).strip()

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f: return yaml.safe_load(f)

def require_columns(df: pd.DataFrame, cols: List[str], where: str):
    missing = [c for c in cols if c not in df.columns]
    if missing: raise ValueError(f"[{where}] Missing: {missing}. Available: {list(df.columns)}")

def build_id_map(values: List[str]) -> Dict[str, int]:
    return {v:i for i,v in enumerate(sorted(set(values)))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config); logger = configure_logger("./logs")
    paths = cfg["paths"]; text_col = cfg["text_col"]; rule_col = cfg["rule_col"]
    subreddit_col = cfg["subreddit_col"]; target_col = cfg["target_col"]
    train_csv = paths["train_csv"]
    if not os.path.exists(train_csv): logger.error(f"train_csv not found: {train_csv}"); raise SystemExit(1)
    df = pd.read_csv(train_csv)
    require_columns(df, [text_col, rule_col, subreddit_col, target_col,
                         "positive_example_1","positive_example_2","negative_example_1","negative_example_2"], "train_csv")
    logger.info(f"Loaded train: shape={df.shape}")
    logger.info(f"Null rate {text_col}: {df[text_col].isna().mean():.4f}")
    if cfg.get("normalization", {}).get("enable", True):
        df["body_norm"] = df[text_col].fillna("").map(soft_normalize)
    else: df["body_norm"] = df[text_col].fillna("")
    rule2id = build_id_map(df[rule_col].astype(str).tolist())
    subreddit2id = build_id_map(df[subreddit_col].astype(str).tolist())
    df["rule_id"] = df[rule_col].map(rule2id); df["subreddit_id"] = df[subreddit_col].map(subreddit2id)
    with open(paths["rules_json"], "w", encoding="utf-8") as f:
        json.dump({"rule2id": rule2id, "id2rule": {v:k for k,v in rule2id.items()}}, f, ensure_ascii=False, indent=2)
    with open(paths["subreddits_json"], "w", encoding="utf-8") as f:
        json.dump({"subreddit2id": subreddit2id, "id2subreddit": {v:k for k,v in subreddit2id.items()}}, f, ensure_ascii=False, indent=2)
    exemplars = {}
    def filt(s): 
        if not isinstance(s, str): return False
        s=s.strip(); return 10 <= len(s) <= 600
    for rule_text, g in df.groupby(rule_col):
        pos, neg = [], []
        for c in ["positive_example_1","positive_example_2"]: pos += g[c].dropna().astype(str).tolist()
        for c in ["negative_example_1","negative_example_2"]: neg += g[c].dropna().astype(str).tolist()
        pos = sorted(set([s.strip() for s in pos if filt(s)]))
        neg = sorted(set([s.strip() for s in neg if filt(s)]))
        exemplars[rule_text] = {"positive": pos, "negative": neg}
    with open(paths["exemplars_json"], "w", encoding="utf-8") as f:
        json.dump(exemplars, f, ensure_ascii=False, indent=2)
    out_cols = (["row_id"] if "row_id" in df.columns else []) + [text_col,"body_norm",rule_col,"rule_id",subreddit_col,"subreddit_id",target_col]
    df[out_cols].to_csv(paths["processed_train_csv"], index=False,encoding="utf-8-sig")
    logger.info("Preprocess done.")
if __name__ == "__main__": main()
