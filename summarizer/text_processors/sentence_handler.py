from typing import List

from spacy.lang.en import English
from spacy.language import Language
from summarizer.text_processors.sentence_abc import SentenceABC


class SentenceHandler(SentenceABC):
    """Basic Sentence Handler."""

    def __init__(self, language: Language = English):
        """
        Base Sentence Handler with Spacy support.

        :param language: Determines the language to use with spacy.
        """
        nlp = language()

        is_spacy_3 = False
        try:
            # Supports spacy 2.0
            nlp.add_pipe(nlp.create_pipe('sentencizer'))
        except Exception:
            # Supports spacy 3.0
            nlp.add_pipe("sentencizer")
            is_spacy_3 = True

        super().__init__(nlp, is_spacy_3)

    def process(
        self, body: str, min_length: int = 40, max_length: int = 600
    ) -> List[str]:
        """
        Processes the content sentences.

        :param body: The raw string body to process
        :param min_length: Minimum length that the sentences must be
        :param max_length: Max length that the sentences mus fall under
        :return: Returns a list of sentences.
        """
        doc = self.nlp(body)
        return self.sentence_processor(doc, min_length, max_length)
