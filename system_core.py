

"""
system_core.py
=======
AutoModerationSystem - orchestrates language detection,
abuse classification, and user/NER detection for chat moderation.
"""

from language_detector  import LanguageDetector
from abuse_classifier   import AbuseClassifier
from user_detector      import UserDetector


class AutoModerationSystem:

    _MODEL_TYPES = ["TF_IDF+SVM", "XLM_RoBERTa"]

    def __init__(
        self,
        languages:       list[str] = ["en", "es", "fr", "ru"],
        model_type:      str = "XLM_RoBERTa",
        abuse_threshold: float = 0.3,
        lang_threshold:  float = 0.25
    ):
        """
        Args:
            languages:       language codes the detector will recognise.
                             Texts in other languages are flagged as unknown.
            model_type:      "XLM_RoBERTa" or "TF_IDF+SVM"
            abuse_threshold: probability cutoff for a label to be considered active.
        """
        print("Initialising AutoModerationSystem...")
        self.lang_detector    = LanguageDetector(languages)
        self.abuse_classifier = AbuseClassifier(
            model_type=model_type,
            threshold=abuse_threshold,
        )
        self.user_detector = UserDetector()
        self.lang_threshold = lang_threshold
        print("System ready.\n")

    # ─────────────────────────────────────────
    #  Core processing
    # ─────────────────────────────────────────

    def process_single(self, text: str) -> dict:
        """
        Run the full moderation pipeline on one message.

        Returns a result dict always containing:
            {
              "text":            original message,
              "lang":            detected language code or "unknown",
              "unknown_lang":    bool,
              "is_toxic":        bool  (False when lang is unknown),
              "abuse_labels":    list of active label names,
              "users_mentioned": list of detected usernames/persons,
            }

        User detection only runs when abuse is detected - saves compute
        on the majority of clean messages.
        """
        result = {
            "text":             text,
            "lang":             None,
            "unknown_lang":     False,
            "is_toxic":         False,
            "abuse_labels":     [],
            "users_mentioned":  [],
        }

        # ── C1: Language detection ────────────────────────────────────────
        lang, confidence = self.lang_detector.detect(text)

        if lang == LanguageDetector._unknownLang or confidence < self.lang_threshold:
            result["lang"]         = "unknown"
            result["unknown_lang"] = True
            return result           # skip further processing for unknown languages

        result["lang"] = lang

        # ── C2: Abuse classification ──────────────────────────────────────
        abuse = self.abuse_classifier.detect(text)
        result["is_toxic"]     = abuse["is_toxic"]
        result["abuse_labels"] = [l for l, v in abuse["labels"].items() if v]

        # ── C3: User detection - only when abuse is found ─────────────────
        if abuse["is_toxic"]:
            users, _ = self.user_detector.detect(text)
            result["users_mentioned"] = users if users else []

        return result

    def process_batch(self, texts: list[str] | str) -> tuple[list[dict], list[str]]:
        """
        Run the pipeline on a list of messages.

        Args:
            texts: a single string or a list of strings.

        Returns:
            (results, unknown_bag) where:
                results:     list of result dicts for processable messages
                unknown_bag: list of raw texts whose language was not recognised
        """
        if isinstance(texts, str):
            texts = [texts]

        results     = []
        unknown_bag = []

        for text in texts:
            result = self.process_single(text)
            if result["unknown_lang"]:
                unknown_bag.append(text)
            else:
                results.append(result)

        return results, unknown_bag

    # ─────────────────────────────────────────
    #  Output
    # ─────────────────────────────────────────

    @staticmethod
    def print_results(results: list[dict] | dict) -> None:
        """Pretty-print moderation results to terminal."""
        if isinstance(results, dict):
            results = [results]

        if not results:
            print("  No results to display.")
            return

        col = 60
        print("\n" + "═" * col)
        print("  MODERATION RESULTS")
        print("═" * col)

        for i, r in enumerate(results, 1):
            status = "TOXIC" if r["is_toxic"] else "CLEAN"
            labels = ", ".join(r["abuse_labels"]) if r["abuse_labels"] else "none"
            users  = ", ".join(r["users_mentioned"]) if r["users_mentioned"] else "none"

            print(f"\n  [{i}] {r['text']!r}")
            print(f"      Status   : {status}")
            print(f"      Language : {r['lang']}")
            print(f"      Labels   : {labels}")
            print(f"      Users    : {users}")

        print("\n" + "═" * col + "\n")

    @staticmethod
    def print_unknown(unknown_bag: list[str]) -> None:
        """Report messages that were skipped due to unrecognised language."""
        if not unknown_bag:
            return
        print(f"\n  FAIL: {len(unknown_bag)} message(s) skipped (unknown language):")
        for text in unknown_bag:
            print(f"    - {text!r}")
        print()


    