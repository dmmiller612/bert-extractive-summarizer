from typing import List

from spacy.language import Language


class SentenceABC:
    """Parent Class for sentence processing."""

    def __init__(self, nlp: Language):
        """
        Base Sentence Handler with Spacy support.

        :param nlp: NLP Pipeline.
        """
        self.nlp = nlp

    def sentence_processor(
        self, doc, min_length: int = 40, max_length: int = 600
    ) -> List[str]:
        """
        Processes a given spacy document and turns them into sentences.

        :param doc: The document to use from spacy.
        :param min_length: The minimum length a sentence should be to be considered.
        :param max_length: The maximum length a sentence should be to be considered.
        :return: Sentences.
        """
        to_return = []

        for c in doc.sents:
            if max_length > len(c.text.strip()) > min_length:
                to_return.append(c.text.strip())

        return to_return

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
        raise NotImplementedError()

    def __call__(
        self, body: str, min_length: int = 40, max_length: int = 600
    ) -> List[str]:
        """
        Processes the content sentences.

        :param body: The raw string body to process
        :param min_length: Minimum length that the sentences must be
        :param max_length: Max length that the sentences mus fall under
        :return: Returns a list of sentences.
        """
        return self.process(body, min_length, max_length)
