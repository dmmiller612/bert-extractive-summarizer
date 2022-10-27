#  Updated to use latest Spacy 3 coreference model

from typing import List

import spacy
import re
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
        Coreference handler. Updated to work with spacy > 3.0.

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
        reindex = []
        for chain in doc.spans:
            for idx, span in enumerate(doc.spans[chain]):
                if idx > 0:
                    reindex.append([span.start_char, span.end_char, doc.spans[chain][0].text, span.text])

        for span in sorted(reindex, key=lambda x: x[0]):
            antecedent = span[2]
            coreferent = span[3]
            antecedent_is_possessive = re.match(r".+['\u2019]s$", antecedent)
            coreferent_is_possessive = re.match(r"^([Ii]ts|[Hh]is|[Hh]er|[Tt]heir|.+['\u2019]s)$", coreferent)
            # Add possessive to antecedent if resolving in a possessive context
            if coreferent_is_possessive and not antecedent_is_possessive:
                    antecedent = antecedent + "\u2019s"
            # Remove possessive from resolved antecedent if not resolving in a possessive context
            elif antecedent_is_possessive and not coreferent_is_possessive:
                antecedent = re.sub(r"['\u2019]s$", "", antecedent)
            resolved_text = resolved_text[0:span[0] + offset] + antecedent + resolved_text[span[1] + offset:]
            offset += len(antecedent) - (span[1] - span[0])

        result_sents = SentenceHandler().process(resolved_text, min_length=min_length)
        return result_sents

