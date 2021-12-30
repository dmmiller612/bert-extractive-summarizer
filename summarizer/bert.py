from functools import partial
from typing import List, Optional, Union

from transformers import (AlbertModel, AlbertTokenizer, BartModel, BigBirdModel, BigBirdTokenizer,
                          BartTokenizer, BertModel, BertTokenizer,
                          CamembertModel, CamembertTokenizer, CTRLModel,
                          CTRLTokenizer, DistilBertModel, DistilBertTokenizer,
                          GPT2Model, GPT2Tokenizer, LongformerModel,
                          LongformerTokenizer, OpenAIGPTModel,
                          OpenAIGPTTokenizer, PreTrainedModel,
                          PreTrainedTokenizer, RobertaModel, RobertaTokenizer,
                          TransfoXLModel, TransfoXLTokenizer, XLMModel,
                          XLMTokenizer, XLNetModel, XLNetTokenizer)

from summarizer.summary_processor import SummaryProcessor
from summarizer.text_processors.sentence_handler import SentenceHandler
from summarizer.transformer_embeddings.bert_embedding import BertEmbedding


class BertSummarizer(SummaryProcessor):
    """Summarizer based on the BERT model."""

    def __init__(
        self,
        model: Optional[str] = 'bert-large-uncased',
        custom_model: PreTrainedModel = None,
        custom_tokenizer: PreTrainedTokenizer = None,
        hidden: Union[List[int], int] = -2,
        reduce_option: str = 'mean',
        sentence_handler: SentenceHandler = SentenceHandler(),
        random_state: int = 12345,
        hidden_concat: bool = False,
        gpu_id: int = 0,
    ):
        """
        This is the parent Bert Summarizer model. New methods should implement this class.

        :param model: This parameter is associated with the inherit string parameters from the transformers library.
        :param custom_model: If you have a pre-trained model, you can add the model class here.
        :param custom_tokenizer: If you have a custom tokenizer, you can add the tokenizer here.
        :param hidden: This signifies which layer(s) of the BERT model you would like to use as embeddings.
        :param reduce_option: Given the output of the bert model, this param determines how you want to reduce results.
        :param sentence_handler: The handler to process sentences. If want to use coreference, instantiate and pass.
        CoreferenceHandler instance
        :param random_state: The random state to reproduce summarizations.
        :param hidden_concat: Whether or not to concat multiple hidden layers.
        :param gpu_id: GPU device index if CUDA is available.
        """
        model = BertEmbedding(model, custom_model, custom_tokenizer, gpu_id)
        model_func = partial(model, hidden=hidden, reduce_option=reduce_option, hidden_concat=hidden_concat)
        super().__init__(model_func, sentence_handler, random_state)


class Summarizer(BertSummarizer):

    def __init__(
        self,
        model: str = 'bert-large-uncased',
        custom_model: PreTrainedModel = None,
        custom_tokenizer: PreTrainedTokenizer = None,
        hidden: Union[List[int], int] = -2,
        reduce_option: str = 'mean',
        sentence_handler: SentenceHandler = SentenceHandler(),
        random_state: int = 12345,
        hidden_concat: bool = False,
        gpu_id: int = 0,
    ):
        """
        This is the main Bert Summarizer class.

        :param model: This parameter is associated with the inherit string parameters from the transformers library.
        :param custom_model: If you have a pre-trained model, you can add the model class here.
        :param custom_tokenizer: If you have a custom tokenizer, you can add the tokenizer here.
        :param hidden: This signifies which layer of the BERT model you would like to use as embeddings.
        :param reduce_option: Given the output of the bert model, this param determines how you want to reduce results.
        :param random_state: The random state to reproduce summarizations.
        :param hidden_concat: Whether or not to concat multiple hidden layers.
        :param gpu_id: GPU device index if CUDA is available. 
        """

        super(Summarizer, self).__init__(
            model, custom_model, custom_tokenizer, hidden, reduce_option, sentence_handler, random_state, hidden_concat,
            gpu_id
        )


class TransformerSummarizer(BertSummarizer):
    """
    Newer style that has keywords for models and tokenizers, but allows the user to change the type.
    """

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
        hidden: Union[List[int], int] = -2,
        reduce_option: str = 'mean',
        sentence_handler: SentenceHandler = SentenceHandler(),
        random_state: int = 12345,
        hidden_concat: bool = False,
        gpu_id: int = 0,
    ):
        """
        :param transformer_type: The Transformer type, such as Bert, GPT2, DistilBert, etc.
        :param transformer_model_key: The transformer model key. This is the directory for the model.
        :param transformer_tokenizer_key: The transformer tokenizer key. This is the tokenizer directory.
        :param hidden: The hidden output layers to use for the summarization.
        :param reduce_option: The reduce option, such as mean, max, min, median, etc.
        :param sentence_handler: The sentence handler class to process the raw text.
        :param random_state: The random state to use.
        :param hidden_concat: Deprecated hidden concat option.
        :param gpu_id: GPU device index if CUDA is available. 
        """
        try:
            self.MODEL_DICT['Roberta'] = (RobertaModel, RobertaTokenizer)
            self.MODEL_DICT['Albert'] = (AlbertModel, AlbertTokenizer)
            self.MODEL_DICT['Camembert'] = (CamembertModel, CamembertTokenizer)
            self.MODEL_DICT['Bart'] = (BartModel, BartTokenizer)
            self.MODEL_DICT['Longformer'] = (LongformerModel, LongformerTokenizer)
            self.MODEL_DICT['BigBird'] = (BigBirdModel, BigBirdTokenizer)
        except Exception:
            pass  # older transformer version

        model_clz, tokenizer_clz = self.MODEL_DICT[transformer_type]
        model = model_clz.from_pretrained(
            transformer_model_key, output_hidden_states=True)

        tokenizer = tokenizer_clz.from_pretrained(
            transformer_tokenizer_key if transformer_tokenizer_key is not None else transformer_model_key
        )

        super().__init__(
            None, model, tokenizer, hidden, reduce_option, sentence_handler, random_state, hidden_concat, gpu_id
        )
