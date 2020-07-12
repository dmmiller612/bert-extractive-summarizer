# removed previous import and related functionality since it's just a blank language model,
#  while neuralcoref requires passing pretrained language model via spacy.load()

import neuralcoref
import spacy

from summarizer.sentence_handler import SentenceHandler


class CoreferenceHandler(SentenceHandler):

    def __init__(self, spacy_model: str = 'en_core_web_sm', greedyness: float = 0.45):
        self.nlp = spacy.load(spacy_model)
        neuralcoref.add_to_pipe(self.nlp, greedyness=greedyness)

    def process(self, body: str, min_length: int = 40, max_length: int = 600):
        """
        Processes the content sentences.

        :param body: The raw string body to process
        :param min_length: Minimum length that the sentences must be
        :param max_length: Max length that the sentences mus fall under
        :return: Returns a list of sentences.
        """
        doc = self.nlp(body)._.coref_resolved
        doc = self.nlp(doc)
        return [c.string.strip() for c in doc.sents if max_length > len(c.string.strip()) > min_length]
