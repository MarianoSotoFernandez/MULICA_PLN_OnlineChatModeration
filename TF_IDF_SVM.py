

"""
TF_IDF_SVM.py
=============
TF-IDF (word + char n-grams) + OneVsRest LinearSVC classifier.

Public API
----------
    build_pipeline()    --> sklearn Pipeline
    train(X, Y)         --> fitted Pipeline   (also saves to disk)
    predict(pipeline, X)--> np.ndarray (N, 6) binary predictions
    load(path)          --> fitted Pipeline loaded from disk

Standalone usage
----------------
    from TF_IDF_SVM import train, predict
"""

import os
import joblib
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.calibration import CalibratedClassifierCV

from utils import clean_text


# ─────────────────────────────────────────────────────────────────────────────
#  Defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH = "./models/tfidf_models/tfidf_svm_model.joblib"

# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline definition
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    Build the TF-IDF + LinearSVC multi-label pipeline.

    Feature union:
        word n-grams (1–2): captures phrases like "kill yourself"
        char n-grams (3–5): catches obfuscated abuse like "f**k", "a55hole"

    Classifier:
        OneVsRestClassifier - one independent binary SVC per label.
        CalibratedClassifierCV - wraps LinearSVC to produce probabilities
        class_weight="balanced" - compensates for the ~10 % toxic rate.
    """
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=100_000,
        sublinear_tf=True,
        strip_accents="unicode",
        min_df=3,
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=100_000,
        sublinear_tf=True,
        strip_accents="unicode",
        min_df=3,
    )
    features = FeatureUnion([("word", word_tfidf), ("char", char_tfidf)])
    svc = CalibratedClassifierCV(
        LinearSVC(C=0.3, max_iter=2000, class_weight="balanced"),
        cv=3,
    )
    return Pipeline([("features", features), ("clf", OneVsRestClassifier(svc))])


# ─────────────────────────────────────────────────────────────────────────────
#  Train
# ─────────────────────────────────────────────────────────────────────────────

def train(
    X_train: list[str],
    Y_train: np.ndarray,
    model_path: str = DEFAULT_MODEL_PATH,
) -> Pipeline:
    """
    Fit the pipeline on training data and save it to disk.

    Args:
        X_train:    list of cleaned text strings
        Y_train:    float32 array of shape (N, 6) with binary labels
        model_path: path to save the fitted pipeline (.joblib)

    Returns:
        Fitted sklearn Pipeline.
    """
    pipeline = build_pipeline()
    print("  Fitting TF-IDF + SVM pipeline...")
    pipeline.fit(X_train, Y_train)
    joblib.dump(pipeline, model_path)
    print(f"  Model saved --> {model_path}")
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
#  Predict
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    pipeline: Pipeline,
    X: list[str],
) -> np.ndarray:
    """
    Generate binary multi-label predictions.

    Args:
        pipeline: fitted sklearn Pipeline (from train() or load())
        X:        list of cleaned text strings

    Returns:
        int array of shape (N, 6) - one binary column per label.
    """
    return pipeline.predict(X)


# ─────────────────────────────────────────────────────────────────────────────
#  Persistence
# ─────────────────────────────────────────────────────────────────────────────

def load(model_path: str = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load a previously saved pipeline from disk."""
    return joblib.load(model_path)


# ─────────────────────────────────────────────────────────────────────────────
#  Inference wrapper (optional standalone use)
# ─────────────────────────────────────────────────────────────────────────────

class TFIDFClassifier:
    """
    Thin inference wrapper for production use.
    """

    LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    def __init__(
        self,
        model_path: str   = DEFAULT_MODEL_PATH,
        threshold:  float = 0.3,
    ):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"TF-IDF model not found at '{model_path}'.\n"
                f"Train it first with: python train_and_evaluate.py"
            )
        self.threshold = threshold
        self.pipeline  = load(model_path)

    def predict_single(self, message: str) -> dict:
        cleaned = clean_text(message)
        proba   = self.pipeline.predict_proba([cleaned])[0]
        scores  = dict(zip(self.LABELS, proba.round(4).tolist()))
        labels  = {l: bool(s >= self.threshold) for l, s in scores.items()}
        return {
            "message":  message,
            "is_toxic": any(labels.values()),
            "labels":   labels,
            "scores":   scores,
        }

    def predict_batch(self, messages: list[str]) -> list[dict]:
        return [self.predict_single(m) for m in messages]
