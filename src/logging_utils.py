import os, sys, logging
from transformers.utils import logging as hf_logging
def configure_logger(log_dir: str, name: str = "jigsaw_w1"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "run.log")
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG); logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8"); fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    try: hf_logging.set_verbosity_warning()
    except Exception: pass
    return logger
