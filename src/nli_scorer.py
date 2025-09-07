import os, argparse, pandas as pd
from transformers import pipeline
from typing import List
from src.logging_utils import configure_logger

def build_hypotheses(rule_text: str, templates: List[str]) -> List[str]:
    return [t.format(rule=rule_text) for t in templates]

def compute_nli_scores(df, text_col, rule_col, model_name, templates, batch_size, logger):
    clf = pipeline("zero-shot-classification", model=model_name, device_map="auto", truncation=True)
    scores = []
    for i, row in df.iterrows():
        body, rule = str(row[text_col]), str(row[rule_col])
        hyps = build_hypotheses(rule, templates)
        vals = []
        for h in hyps:
            res = clf(body, candidate_labels=[h], multi_label=True)
            vals.append(float(res["scores"][0]))
        scores.append(sum(vals)/len(vals))
        if (i+1) % 100 == 0: logger.info(f"NLI {i+1}/{len(df)}")
    return scores
