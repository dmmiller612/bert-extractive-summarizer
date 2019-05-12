from summarizer.BertParent import BertParent
from typing import List
from summarizer.ClusterFeatures import ClusterFeatures
from abc import abstractmethod
import neuralcoref
from spacy.lang.en import English


class ModelProcessor(object):

    def __init__(self, model='bert-large-uncased',
                 vector_size: int = None,
                 hidden: int=-2,
                 reduce_option: str = 'mean',
                 greedyness: float=0.45):
        self.model = BertParent(model)
        self.hidden = hidden
        self.vector_size = vector_size
        self.reduce_option = reduce_option
        self.nlp = English()
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
        neuralcoref.add_to_pipe(self.nlp, greedyness=greedyness)

    def process_content_sentences(self, body: str, min_length=40, max_length=600) -> List[str]:
        doc = self.nlp(body)._.coref_resolved
        doc = self.nlp(doc)
        return [c.string.strip() for c in doc.sents
                if len(c.string.strip()) > min_length and len(c.string.strip()) < max_length]

    @abstractmethod
    def run_clusters(self, content: List[str], ratio=0.2, algorithm='kmeans', use_first: bool=True) -> List[str]:
        raise NotImplementedError("Must Implement run_clusters")

    def run(self, body: str, ratio: float=0.2, min_length: int=40, max_length: int=600,
            use_first: bool=True, algorithm='kmeans') -> str:
        sentences = self.process_content_sentences(body, min_length, max_length)
        res = self.run_clusters(sentences, ratio, algorithm, use_first)
        return ' '.join(res)

    def __call__(self, body: str, ratio: float=0.2, min_length: int=40, max_length: int=600,
                 use_first: bool=True, algorithm='kmeans') -> str:
        return self.run(body, ratio, min_length, max_length)


class SingleModel(ModelProcessor):

    def __init__(self, model='bert-large-uncased',
                 vector_size: int = None,
                 hidden: int=-2,
                 reduce_option: str = 'mean',
                 greedyness: float=0.45):
        super(SingleModel, self).__init__(model, vector_size, hidden, reduce_option, greedyness)

    def run_clusters(self, content: List[str], ratio=0.2, algorithm='kmeans', use_first: bool= True) -> List[str]:
        hidden = self.model(content, self.hidden, self.reduce_option)
        hidden_args = ClusterFeatures(hidden, algorithm).cluster(ratio)
        if use_first:
            if hidden_args[0] != 0:
                hidden_args.insert(0,0)
        return [content[j] for j in hidden_args]

