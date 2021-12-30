from typing import List

from spacy.language import Language


class SentenceABC:
    """Parent Class for sentence processing."""

    def __init__(self, nlp: Language, is_spacy_3: bool):
        """
        Base Sentence Handler with Spacy support.

        :param nlp: NLP Pipeline.
        :param is_spacy_3: Whether or not we are using spacy 3.
        """
        self.nlp = nlp
        self.is_spacy_3 = is_spacy_3

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

                if self.is_spacy_3:
                    to_return.append(c.text.strip())
                else:
                    to_return.append(c.string.strip())

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
