"""
train_and_evaluate.py
=====================
Orchestrator: loads data, trains both models on the same splits,
evaluates on the same held-out test set, and prints a full metric report.

Usage examples
--------------
    python train_and_evaluate.py                     # full dataset + both models
    python train_and_evaluate.py --sample 10000      # 10000 data   + both models
    python train_and_evaluate.py --skip-tfidf        # full data    + XLM-R only
    python train_and_evaluate.py --skip-xlmr         # full data    + TF-IDF only

Data split
----------
    70 % train  |  15 % validation  |  15 % test
    Both models are evaluated on the exact same test set for a fair comparison.
    The validation split is used only by XLM-RoBERTa for checkpoint selection.
"""

import time
import argparse
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
import torch

import utils
import TF_IDF_SVM
import XLM_RoBERTa

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI arguments
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate TF-IDF+SVM and XLM-RoBERTa toxic classifiers."
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of rows to sample from the dataset (default: full ~160k).",
    )
    parser.add_argument(
        "--skip-tfidf",
        action="store_true",
        help="Skip TF-IDF + SVM training and evaluation.",
    )
    parser.add_argument(
        "--skip-xlmr",
        action="store_true",
        help="Skip XLM-RoBERTa training and evaluation.",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default=utils.TRAIN_FILE,
        help="Path to train.csv (default: ./data/train.csv).",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_splits(
    train_path: str,
    sample_size: int | None,
) -> tuple[list, list, list, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data and produce train / val / test splits.

    Stratified on the 'toxic' column to preserve class ratios.

    Returns:
        X_train, X_val, X_test, Y_train, Y_val, Y_test
    """
    df = utils.load_data(path=train_path, sample_size=sample_size)

    print(f"\n  Rows loaded : {len(df):,}")
    print(f"  Label distribution:\n{df[utils.LABELS].sum().to_frame('count').T.to_string()}")

    X = df["clean"].tolist()
    Y = df[utils.LABELS].values.astype("float32")

    # 70 / 15 / 15
    X_tv, X_test, Y_tv, Y_test = train_test_split(
        X, Y, test_size=0.15, random_state=42, stratify=Y[:, 0]
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_tv, Y_tv,
        test_size=0.15 / 0.85,   # keeps overall test fraction at 15 %
        random_state=42,
        stratify=Y_tv[:, 0],
    )

    print(f"\n  Split sizes:")
    print(f"    Train : {len(X_train):,}")
    print(f"    Val   : {len(X_val):,}  (XLM-R checkpoint selection only)")
    print(f"    Test  : {len(X_test):,}  (shared evaluation set)")

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    
    print("\n── GPU status ───────────────────────────────")
    print(f"  CUDA available : {torch.cuda.is_available()}")
    print(f"  Device         : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  CUDA version   : {torch.version.cuda}")
    print("─────────────────────────────────────────────\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    utils.print_header("LOADING DATA")
    X_train, X_val, X_test, Y_train, Y_val, Y_test = prepare_splits(
        train_path=args.train_path,
        sample_size=args.sample,
    )

    results: dict[str, np.ndarray] = {}   # model_name --> Y_pred on test set

    # ── Model 1: TF-IDF + SVM ────────────────────────────────────────────────
    if not args.skip_tfidf:
        utils.print_header("MODEL 1 — TF-IDF + LinearSVC")
        t0       = time.time()
        pipeline = TF_IDF_SVM.train(X_train, Y_train)
        pred     = TF_IDF_SVM.predict(pipeline, X_test)
        elapsed  = time.time() - t0

        utils.print_metrics("TF-IDF + LinearSVC", Y_test, pred, elapsed)
        results["TF-IDF + SVM"] = pred

    # ── Model 2: XLM-RoBERTa ─────────────────────────────────────────────────
    if not args.skip_xlmr:
        utils.print_header("MODEL 2 — XLM-RoBERTa (fine-tuned)")
        t0               = time.time()
        model, tokenizer = XLM_RoBERTa.train(X_train, Y_train, X_val, Y_val)

        print("\n  Running inference on test set...")
        pred    = XLM_RoBERTa.predict(model, tokenizer, X_test)
        elapsed = time.time() - t0

        utils.print_metrics("XLM-RoBERTa (fine-tuned)", Y_test, pred, elapsed)
        results["XLM-RoBERTa"] = pred

    # ── Side-by-side comparison ───────────────────────────────────────────────
    if len(results) > 1:
        utils.print_comparison(Y_test, results)
    elif len(results) == 0:
        print("\n  Nothing to evaluate — both models were skipped.")


if __name__ == "__main__":
    main()
