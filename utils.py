"""
utils.py
========
Shared constants, text preprocessing, data loading,
and metric reporting used by both model modules.
"""

import os
import re
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

LABELS     = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
NUM_LABELS = len(LABELS)
TRAIN_FILE = "./data/train.csv"


# ─────────────────────────────────────────────────────────────────────────────
#  Text preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Lightweight cleaning for toxic comment text.
    Keeps punctuation (!, ?) since it carries signal for abuse detection.
    """
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)      # remove URLs
    text = re.sub(r"<.*?>", " ", text)                      # strip HTML tags
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)              # Reduction ("loooser" --> "looser")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(
    path: str = TRAIN_FILE,
    sample_size: int | None = None,
) -> pd.DataFrame:
    """
    Load Jigsaw CSV from disk and apply text cleaning.

    Args:
        path:        path to train.csv (default: ./data/train.csv)
        sample_size: number of rows to sample (None = full dataset)

    Returns:
        DataFrame with original columns plus a 'clean' text column.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            f"Download train.csv from:\n"
            f"  https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data"
        )

    df = pd.read_csv(path)

    if sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    df["clean"] = df["comment_text"].map(clean_text)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Terminal reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_header(title: str, width: int = 62) -> None:
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def print_metrics(
    model_name: str,
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    elapsed: float,
) -> None:
    """
    Print per-label and aggregate metrics to terminal.

    Metrics reported:
        - Subset Accuracy  (all labels must match for a sample to count)
        - Macro / Micro F1
        - Macro Precision
        - Macro Recall
        - Per-label precision / recall / f1 / support
    """
    acc      = accuracy_score(Y_true, Y_pred)
    macro_f1 = f1_score(Y_true, Y_pred, average="macro",  zero_division=0)
    micro_f1 = f1_score(Y_true, Y_pred, average="micro",  zero_division=0)
    macro_p  = precision_score(Y_true, Y_pred, average="macro",  zero_division=0)
    macro_r  = recall_score(Y_true, Y_pred, average="macro",  zero_division=0)

    sep = "─" * 62
    print(f"\n{sep}")
    print(f"  Model  : {model_name}")
    print(f"  Time   : {elapsed:.1f}s")
    print(sep)
    print(f"  Subset Accuracy  : {acc:.4f}")
    print(f"  Macro  F1        : {macro_f1:.4f}")
    print(f"  Micro  F1        : {micro_f1:.4f}")
    print(f"  Macro  Precision : {macro_p:.4f}")
    print(f"  Macro  Recall    : {macro_r:.4f}")
    print(sep)
    print(f"\n  Per-label breakdown:\n")
    print(classification_report(Y_true, Y_pred, target_names=LABELS, zero_division=0))


def print_comparison(
    Y_true: np.ndarray,
    results: dict[str, np.ndarray],   # {model_name: Y_pred}
) -> None:
    """
    Print a side-by-side metric table for all evaluated models.

    Args:
        Y_true:  ground-truth multi-label array  (N, 6)
        results: mapping of model name --> predicted array (N, 6)
    """
    print_header("SIDE-BY-SIDE COMPARISON")

    metric_fns = {
        "Subset Accuracy": accuracy_score,
        "Macro F1":        lambda y, p: f1_score(y, p, average="macro",  zero_division=0),
        "Micro F1":        lambda y, p: f1_score(y, p, average="micro",  zero_division=0),
        "Macro Precision": lambda y, p: precision_score(y, p, average="macro", zero_division=0),
        "Macro Recall":    lambda y, p: recall_score(y, p, average="macro",    zero_division=0),
    }

    col_w      = 22
    model_col  = 13
    names      = list(results.keys())

    # Header row
    header = f"  {'Metric':<{col_w}}" + "".join(f"{n:>{model_col}}" for n in names)
    print(header)
    print(f"  {'─' * (col_w + model_col * len(names))}")

    for metric_name, fn in metric_fns.items():
        row = f"  {metric_name:<{col_w}}"
        for name, Y_pred in results.items():
            row += f"{fn(Y_true, Y_pred):>{model_col}.4f}"
        print(row)

    print(f"  {'─' * (col_w + model_col * len(names))}\n")
