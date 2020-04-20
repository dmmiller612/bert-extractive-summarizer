from summarizer.bert_parent import BertParent
from summarizer.cluster_features import ClusterFeatures
from summarizer.sentence_handler import SentenceHandler
from typing import List
from abc import abstractmethod
import numpy as np
from transformers import *


class ModelProcessor(object):

    def __init__(
        self,
        model: str = 'bert-large-uncased',
        custom_model: PreTrainedModel = None,
        custom_tokenizer: PreTrainedTokenizer = None,
        hidden: int = -2,
        reduce_option: str = 'mean',
        sentence_handler: SentenceHandler = SentenceHandler(),
        random_state: int = 12345
    ):
        """
        This is the parent Bert Summarizer model. New methods should implement this class

        :param model: This parameter is associated with the inherit string parameters from the transformers library.
        :param custom_model: If you have a pre-trained model, you can add the model class here.
        :param custom_tokenizer: If you have a custom tokenizer, you can add the tokenizer here.
        :param hidden: This signifies which layer of the BERT model you would like to use as embeddings.
        :param reduce_option: Given the output of the bert model, this param determines how you want to reduce results.
        :param sentence_handler: The handler to process sentences. If want to use coreference, instantiate and pass CoreferenceHandler instance
        :param random_state: The random state to reproduce summarizations.
        """

        np.random.seed(random_state)
        self.model = BertParent(model, custom_model, custom_tokenizer)
        self.hidden = hidden
        self.reduce_option = reduce_option
        self.sentence_handler = sentence_handler
        self.random_state = random_state

    def process_content_sentences(self, body: str, min_length:int = 40, max_length: int = 600) -> List[str]:
        """
        Processes the content sentences with neural coreference.
        :param body: The raw string body to process
        :param min_length: Minimum length that the sentences must be
        :param max_length: Max length that the sentences mus fall under
        :return: Returns a list of sentences with coreference applied.
        """

        doc = self.nlp(body)._.coref_resolved
        doc = self.nlp(doc)
        return [c.string.strip() for c in doc.sents if max_length > len(c.string.strip()) > min_length]

    @abstractmethod
    def run_clusters(
        self,
        content: List[str],
        ratio:float = 0.2,
        algorithm: str = 'kmeans',
        use_first: bool = True
    ) -> List[str]:
        """
        Classes must implement this to run the clusters.
        """
        raise NotImplementedError("Must Implement run_clusters")

    def run(
        self,
        body: str,
        ratio: float = 0.2,
        min_length: int = 40,
        max_length: int = 600,
        use_first: bool = True,
        algorithm: str ='kmeans'
    ) -> str:
        """
        Preprocesses the sentences, runs the clusters to find the centroids, then combines the sentences.

        :param body: The raw string body to process
        :param ratio: Ratio of sentences to use
        :param min_length: Minimum length of sentence candidates to utilize for the summary.
        :param max_length: Maximum length of sentence candidates to utilize for the summary
        :param use_first: Whether or not to use the first sentence
        :param algorithm: Which clustering algorithm to use. (kmeans, gmm)
        :return: A summary sentence
        """
        sentences = self.sentence_handler(body, min_length, max_length)

        if sentences:
            sentences = self.run_clusters(sentences, ratio, algorithm, use_first)

        return ' '.join(sentences)

    def __call__(
        self,
        body: str,
        ratio: float = 0.2,
        min_length: int = 40,
        max_length: int = 600,
        use_first: bool = True,
        algorithm: str = 'kmeans'
    ) -> str:
        """
        (utility that wraps around the run function)

        Preprocesses the sentences, runs the clusters to find the centroids, then combines the sentences.

        :param body: The raw string body to process
        :param ratio: Ratio of sentences to use
        :param min_length: Minimum length of sentence candidates to utilize for the summary.
        :param max_length: Maximum length of sentence candidates to utilize for the summary
        :param use_first: Whether or not to use the first sentence
        :param algorithm: Which clustering algorithm to use. (kmeans, gmm)
        :return: A summary sentence
        """
        return self.run(body, ratio, min_length, max_length, algorithm=algorithm, use_first=use_first)


class SingleModel(ModelProcessor):
    """
    Deprecated for naming sake.
    """

    def __init__(
        self,
        model='bert-large-uncased',
        custom_model: PreTrainedModel = None,
        custom_tokenizer: PreTrainedTokenizer = None,
        hidden: int=-2,
        reduce_option: str = 'mean',
        sentence_handler: SentenceHandler = SentenceHandler(),
        random_state: int=12345
    ):
        super(SingleModel, self).__init__(
            model=model, custom_model=custom_model, custom_tokenizer=custom_tokenizer,
            hidden=hidden, reduce_option=reduce_option,
            sentence_handler=sentence_handler, random_state=random_state
        )

    def run_clusters(self, content: List[str], ratio=0.2, algorithm='kmeans', use_first: bool= True) -> List[str]:
        hidden = self.model(content, self.hidden, self.reduce_option)
        hidden_args = ClusterFeatures(hidden, algorithm, random_state=self.random_state).cluster(ratio)

        if use_first:
            if hidden_args[0] != 0:
                hidden_args.insert(0,0)

        return [content[j] for j in hidden_args]


class Summarizer(SingleModel):

    def __init__(
        self,
        model: str = 'bert-large-uncased',
        custom_model: PreTrainedModel = None,
        custom_tokenizer: PreTrainedTokenizer = None,
        hidden: int = -2,
        reduce_option: str = 'mean',
        sentence_handler: SentenceHandler = SentenceHandler(),
        random_state: int = 12345
    ):
        """
        This is the main Bert Summarizer class.

        :param model: This parameter is associated with the inherit string parameters from the transformers library.
        :param custom_model: If you have a pre-trained model, you can add the model class here.
        :param custom_tokenizer: If you have a custom tokenizer, you can add the tokenizer here.
        :param hidden: This signifies which layer of the BERT model you would like to use as embeddings.
        :param reduce_option: Given the output of the bert model, this param determines how you want to reduce results.
        :param greedyness: associated with the neuralcoref library. Determines how greedy coref should be.
        :param language: Which language to use for training.
        :param random_state: The random state to reproduce summarizations.
        """
        super(Summarizer, self).__init__(
            model, custom_model, custom_tokenizer, hidden, reduce_option, sentence_handler, random_state
        )


class TransformerSummarizer(SingleModel):

    MODEL_DICT = {
        'Bert': (BertModel, BertTokenizer),
        'OpenAIGPT': (OpenAIGPTModel, OpenAIGPTTokenizer),
        'GPT2': (GPT2Model, GPT2Tokenizer),
        'CTRL': (CTRLModel, CTRLTokenizer),
        'TransfoXL': (TransfoXLModel, TransfoXLTokenizer),
        'XLNet': (XLNetModel, XLNetTokenizer),
        'XLM': (XLMModel, XLMTokenizer),
        'DistilBert': (DistilBertModel, DistilBertTokenizer),
    }

    def __init__(
        self,
        transformer_type: str = 'Bert',
        transformer_model_key: str = 'bert-base-uncased',
        transformer_tokenizer_key: str = None,
        hidden: int = -2,
        reduce_option: str = 'mean',
        sentence_handler: SentenceHandler = SentenceHandler(),
        random_state: int = 12345
    ):

        try:
            self.MODEL_DICT['Roberta'] = (RobertaModel, RobertaTokenizer)
            self.MODEL_DICT['Albert'] = (AlbertModel, AlbertTokenizer)
            self.MODEL_DICT['Camembert'] = (CamembertModel, CamembertTokenizer)
        except Exception as e:
            pass # older transformer version

        model_clz, tokenizer_clz = self.MODEL_DICT[transformer_type]
        model = model_clz.from_pretrained(transformer_model_key, output_hidden_states=True)

        tokenizer = tokenizer_clz.from_pretrained(
            transformer_tokenizer_key if transformer_tokenizer_key is not None else transformer_model_key
        )

        super().__init__(
            None, model, tokenizer, hidden, reduce_option, sentence_handler, random_state
        )
