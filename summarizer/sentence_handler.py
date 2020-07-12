from typing import List

from spacy.lang.en import English


class SentenceHandler(object):

    def __init__(self, language=English):
        self.nlp = language()
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

    def process(self, body: str, min_length: int = 40, max_length: int = 600) -> List[str]:
        """
        Processes the content sentences.

        :param body: The raw string body to process
        :param min_length: Minimum length that the sentences must be
        :param max_length: Max length that the sentences mus fall under
        :return: Returns a list of sentences.
        """
        doc = self.nlp(body)
        return [c.string.strip() for c in doc.sents if max_length > len(c.string.strip()) > min_length]

    def __call__(self, body: str, min_length: int = 40, max_length: int = 600) -> List[str]:
        return self.process(body, min_length, max_length)
