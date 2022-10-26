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
        nlp = language(disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
        nlp.add_pipe("sentencizer")
        super().__init__(nlp)

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
