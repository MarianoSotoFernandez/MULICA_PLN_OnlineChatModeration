


"""
abuse_classifier.py
===================
Unified wrapper for the two toxic comment classifiers.
"""

import os
from TF_IDF_SVM import TFIDFClassifier
from XLM_RoBERTa import XLMRClassifier


class AbuseClassifier:

    LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    _MODEL_CLASSES = {
        "TF_IDF+SVM"  : TFIDFClassifier,
        "XLM_RoBERTa" : XLMRClassifier,
    }

    _DEFAULT_PATHS = {
        "TF_IDF+SVM"  : "./models/tfidf_models/tfidf_svm_model.joblib",
        "XLM_RoBERTa" : "./models/xlmroberta_models",   # directory, not a file
    }

    def __init__(
        self,
        model_type: str = "XLM_RoBERTa",
        model_path: str | None = None,
        threshold: float = 0.3,
    ):
        """
        Args:
            model_type: "TF_IDF+SVM" or "XLM_RoBERTa"
            model_path: override the default saved model path/directory.
                        If None, uses the default for the selected model_type.
            threshold:  probability cutoff for positive label (default 0.3)
        """
        if model_type not in self._MODEL_CLASSES:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Choose from: {list(self._MODEL_CLASSES.keys())}"
            )

        self.model_type = model_type
        self.threshold  = threshold
        self.model_path = model_path or self._DEFAULT_PATHS[model_type]

        # Validate path exists (file for TF-IDF, directory for XLM-R)
        path_exists = (
            os.path.isfile(self.model_path)
            if model_type == "TF_IDF+SVM"
            else os.path.isdir(self.model_path)
        )
        if not path_exists:
            raise FileNotFoundError(
                f"Model not found at '{self.model_path}'.\n"
                f"Train it first with: python train_and_evaluate.py"
            )

        cls        = self._MODEL_CLASSES[model_type]
        self.model = cls(model_path=self.model_path, threshold=threshold)
        print(f"  AbuseClassifier ready - model: {self.model_type} | threshold: {threshold}")

    # ─────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────

    def detect(self, text: str) -> dict:
        """
        Classify a single message.

        Returns:
            {
              "message":  original text,
              "is_toxic": bool,
              "labels":   {"toxic": bool, "severe_toxic": bool, ...},
              "scores":   {"toxic": float, ...}   (TF-IDF only; XLM-R omits scores)
            }
        """
        result = self.model.predict_single(text)
        self._print_result(result)
        return result

    def detect_batch(self, texts: list[str]) -> list[dict]:
        """
        Classify a list of messages efficiently.

        Returns:
            List of result dicts (same structure as detect()).
        """
        results = self.model.predict_batch(texts)
        for r in results:
            self._print_result(r)
        return results

    # ─────────────────────────────────────────
    #  Internal helpers
    # ─────────────────────────────────────────

    @staticmethod
    def _print_result(result: dict) -> None:
        status  = "TOXIC" if result["is_toxic"] else "CLEAN"
        active  = [l for l, v in result["labels"].items() if v]
        label_s = ", ".join(active) if active else "none"
        print(f"  [{status}]  {result['message']!r}  -->  {label_s}")