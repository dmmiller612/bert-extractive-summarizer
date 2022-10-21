# removed previous import and related functionality since it's just a blank language model,
#  while neuralcoref requires passing pretrained language model via spacy.load()

from typing import List

import spacy

from summarizer.text_processors.sentence_abc import SentenceABC

# Experimental coref model
DEFAULT_MODEL = "en_coreference_web_trf"


class CoreferenceHandler(SentenceABC):
    """HuggingFace Coreference Handler."""

    def __init__(
        self, spacy_model: str = DEFAULT_MODEL
    ):
        """
        Corefence handler. Only works with spacy < 3.0.

        :param spacy_model: The spacy model to use as default.
        """
        nlp = spacy.load(spacy_model)
        super().__init__(nlp, is_spacy_3=True)

    def process(self, body: str, min_length: int = 40, max_length: int = 600) -> List[str]:
        """
        Processes the content sentences.

        :param body: The raw string body to process
        :param min_length: Minimum length that the sentences must be
        :param max_length: Max length that the sentences mus fall under
        :return: Returns a list of sentences.
        """
        # doc = self.nlp(body)._.coref_resolved
        doc = self.nlp(body)
        return [str(i) for k, v in doc.spans.items() for i in v]
        # Need something like
        # for k, v in doc.spans.items():
        #     for i in v:
        #         print(i, i.start, i.end)

        # return [c.string.strip()
        #         for c in doc.sents
        #         if max_length > len(c.string.strip()) > min_length]
