import argparse, os, json
import pandas as pd
from transformers import pipeline
from logging_utils import configure_logger

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--model", type=str, default="microsoft/deberta-xlarge-mnli")
    args = ap.parse_args()
    import yaml
    with open(args.config,"r",encoding="utf-8") as f: cfg=yaml.safe_load(f)
    logger = configure_logger("./logs")
    paths = cfg["paths"]; text_col=cfg["text_col"]; rule_col=cfg["rule_col"]
    proc = paths["processed_train_csv"]
    if not os.path.exists(proc): logger.error(f"processed_train_csv not found: {proc}"); raise SystemExit(1)
    df = pd.read_csv(proc)
    clf = pipeline("zero-shot-classification", model=args.model, device_map="auto", truncation=True)
    def hyps(rule): return [f"이 댓글은 '{rule}' 규칙을 위반한다.",
                            f"댓글이 '{rule}'를 어긴다.",
                            f"해당 댓글은 '{rule}' 규칙에 저촉된다."]
    probs=[]
    for i,row in df.iterrows():
        scores=[float(clf(str(row[text_col]), candidate_labels=[h], multi_label=True)["scores"][0]) for h in hyps(str(row[rule_col]))]
        probs.append(sum(scores)/len(scores))
        if (i+1)%100==0: logger.info(f"NLI progress {i+1}/{len(df)}")
    out = pd.DataFrame({"row_id": df.get("row_id", pd.Series(range(len(df)))),
                        "rule": df[rule_col],
                        "nli_entail_prob": probs})
    out.to_csv(paths["nli_scores_train"], index=False)
    logger.info(f"Saved -> {paths['nli_scores_train']}")
if __name__=="__main__": main()
