

from lingua import Language, LanguageDetectorBuilder

class LanguageDetector:
    """
    Detector backed by lingua-language-detector.
    """
    
    # Unknown type
    _unknownLang = "unknown"

    # Map lingua's Language enum to ISO 639-1 codes
    _ISO = {
        Language.ENGLISH: "en",
        Language.SPANISH: "es",
        Language.FRENCH: "fr",
        Language.GERMAN: "de",
        Language.PORTUGUESE: "pt",
        Language.ITALIAN: "it",
        Language.DUTCH: "nl",
        Language.RUSSIAN: "ru",
        Language.CHINESE: "zh",
        Language.JAPANESE: "ja",
        Language.ARABIC: "ar",
        Language.POLISH: "pl",
        Language.TURKISH: "tr",
        Language.SWEDISH: "sv",
        Language.DANISH: "da",
    }


    def __init__(self, languages: list[str] | None = None):
        """
        Args:
            languages: optional list of ISO 639-1 codes to limit detection to.
                       Pass None to detect all supported languages.
        
        Example:
            detector = LanguageDetector(languages=["en", "es", "fr"])
        """
        # Reverse lookup: ISO code --> Language enum
        iso_to_lang = {v: k for k, v in self._ISO.items()}

        if languages:
            selected = [iso_to_lang[code] for code in languages if code in iso_to_lang]
        else:
            selected = list(self._ISO.keys())

        self._detector = (
            LanguageDetectorBuilder
            .from_languages(*selected)
            .with_preloaded_language_models()  # faster after first call
            .build()
        )
        self._selected = selected


    def detect(self, text: str) -> tuple[str, float]:
        """
        Returns the most likely (language_code, confidence) for *text*.
        """
        result = self._detector.detect_language_of(text)
        if result is None:
            return (self._unknownLang, 0.0)
        confidence = self._detector.compute_language_confidence(text, result)
        return (self._ISO.get(result, result.name.lower()), round(confidence, 4))


    def detect_all(self, text: str) -> list[tuple[str, float]]:
        """
        Returns all languages with confidence scores, sorted descending.
        """
        results = self._detector.compute_language_confidence_values(text)
        return [
            (self._ISO.get(lang, lang.name.lower()), round(score, 4))
            for lang, score in results
            if score > 0
        ]