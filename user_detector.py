

import spacy

class UserDetector:
    """
    Extracts person mentions from individual game chat messages.
    Designed as a component in a verbal abuse classification pipeline.
    """

    def __init__(self, model: str = "en_core_web_sm"):
        self.nlp = spacy.load(model)

    def detect(self, message: str) -> dict:
        doc = self.nlp(message)

        persons = list(dict.fromkeys(
            ent.text.strip()
            for ent in doc.ents
            if ent.label_ == "PERSON"
        ))

        return [persons, len(persons)]

    def extract_batch(self, messages: list[str]) -> list[dict]:
        return [self.detect(msg) for msg in messages]



