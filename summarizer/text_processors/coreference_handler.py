#  Updated to use latest Spacy 3 coreference model

from typing import List

import spacy

from summarizer.text_processors.sentence_abc import SentenceABC
from summarizer.text_processors.sentence_handler import SentenceHandler

# Experimental coref model
DEFAULT_MODEL = "en_coreference_web_trf"


class CoreferenceHandler(SentenceABC):
    """HuggingFace Coreference Handler."""

    def __init__(
        self, spacy_model: str = DEFAULT_MODEL
    ):
        """
        Corefence handler. Updated to work with spacy > 3.0.

        :param spacy_model: The spacy model to use as default.
        """
        nlp = spacy.load(spacy_model)
        super().__init__(nlp)

    def process(self, body: str, min_length: int = 40, max_length: int = 600) -> List[str]:
        """
        Processes the content sentences.

        :param body: The raw string body to process
        :param min_length: Minimum length that the sentences must be
        :param max_length: Max length that the sentences mus fall under
        :return: Returns a list of sentences.
        """
        doc = self.nlp(body)
        resolved_text = body
        offset = 0
        for chain in doc.spans:
            for idx, span in enumerate(doc.spans[chain]):
                if idx > 0:
                    resolved_text = resolved_text[0:span.start_char + offset] + doc.spans[chain][0].text + resolved_text[span.end_char + offset:]
                    offset += len(doc.spans[chain][0].text) - (span.end_char - span.start_char)
        result_sents = SentenceHandler().process(resolved_text, min_length=min_length)
        return result_sents

