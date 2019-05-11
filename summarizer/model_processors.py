from summarizer.BertParent import BertParent
from typing import List
from summarizer.ClusterFeatures import ClusterFeatures
from nltk import tokenize
from abc import abstractmethod


class ModelProcessor(object):

    def __init__(self, model='bert-large-uncased',
                 vector_size: int = None,
                 hidden: int=-2,
                 reduce_option: str = 'mean'):
        self.model = BertParent(model)
        self.hidden = hidden
        self.vector_size = vector_size
        self.reduce_option = reduce_option

    def process_content_sentences(self, body: str, min_length=40, max_length=600) -> List[str]:
        sentences = tokenize.sent_tokenize(body)
        return [c for c in sentences if len(c) > min_length and len(c) < max_length]

    @abstractmethod
    def run_clusters(self, content: List[str], ratio=0.2, algorithm='kmeans', use_first: bool=True) -> List[str]:
        raise NotImplementedError("Must Implement run_clusters")

    def run(self, body: str, ratio: float=0.2, min_length: int=40, max_length: int=600,
            use_first: bool=True, algorithm='kmeans'):
        sentences = self.process_content_sentences(body, min_length, max_length)
        return self.run_clusters(sentences, ratio, algorithm, use_first)

    def __call__(self, body: str, ratio: float=0.2, min_length: int=40, max_length: int=600,
                 use_first: bool=True, algorithm='kmeans'):
        return self.run(body, ratio, min_length, max_length)


class SingleModelProcessor(ModelProcessor):

    def __init__(self, model='bert-large-uncased',
                 vector_size: int = None,
                 hidden: int=-2,
                 reduce_option: str = 'mean'):
        super(SingleModelProcessor, self).__init__(model, vector_size, hidden, reduce_option)

    def run_clusters(self, content: List[str], ratio=0.2, algorithm='kmeans', use_first: bool= True) -> List[str]:
        hidden = self.model(content, self.hidden, self.reduce_option)
        hidden_args = ClusterFeatures(hidden, algorithm).cluster(ratio)
        if use_first:
            if hidden_args[0] != 0:
                hidden_args.insert(0,0)
        return [content[j] for j in hidden_args]

