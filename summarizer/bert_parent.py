import logging
from typing import List

import numpy as np
import torch
from numpy import ndarray
from transformers import *

logging.basicConfig(level=logging.WARNING)


class BertParent(object):

    """
    Base handler for BERT models.
    """

    MODELS = {
        'bert-base-uncased': (BertModel, BertTokenizer),
        'bert-large-uncased': (BertModel, BertTokenizer),
        'xlnet-base-cased': (XLNetModel, XLNetTokenizer),
        'xlm-mlm-enfr-1024': (XLMModel, XLMTokenizer),
        'distilbert-base-uncased': (DistilBertModel, DistilBertTokenizer),
        'albert-base-v1': (AlbertModel, AlbertTokenizer),
        'albert-large-v1': (AlbertModel, AlbertTokenizer)
    }

    def __init__(
        self,
        model: str,
        custom_model: PreTrainedModel=None,
        custom_tokenizer: PreTrainedTokenizer=None
    ):
        """
        :param model: Model is the string path for the bert weights. If given a keyword, the s3 path will be used
        :param custom_model: This is optional if a custom bert model is used
        :param custom_tokenizer: Place to use custom tokenizer
        """

        base_model, base_tokenizer = self.MODELS.get(model, (None, None))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if custom_model:
            self.model = custom_model.to(self.device)
        else:
            self.model = base_model.from_pretrained(model, output_hidden_states=True).to(self.device)

        if custom_tokenizer:
            self.tokenizer = custom_tokenizer
        else:
            self.tokenizer = base_tokenizer.from_pretrained(model)

        self.model.eval()

    def tokenize_input(self, text: str) -> torch.tensor:
        """
        Tokenizes the text input.

        :param text: Text to tokenize
        :return: Returns a torch tensor
        """
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return torch.tensor([indexed_tokens]).to(self.device)

    def extract_embeddings(
        self,
        text: str,
        hidden: int=-2,
        reduce_option: str ='mean'
    ) -> torch.Tensor:

        """
        Extracts the embeddings for the given text

        :param text: The text to extract embeddings for.
        :param hidden: The hidden layer to use for a readout handler
        :param squeeze: If we should squeeze the outputs (required for some layers)
        :param reduce_option: How we should reduce the items.
        :return: A numpy array.
        """

        tokens_tensor = self.tokenize_input(text)
        pooled, hidden_states = self.model(tokens_tensor)[-2:]

        if -1 > hidden > -12:

            if reduce_option == 'max':
                pooled = hidden_states[hidden].max(dim=1)[0]

            elif reduce_option == 'median':
                pooled = hidden_states[hidden].median(dim=1)[0]

            else:
                pooled = hidden_states[hidden].mean(dim=1)

        return pooled

    def create_matrix(
        self,
        content: List[str],
        hidden: int=-2,
        reduce_option: str = 'mean'
    ) -> ndarray:
        """
        Create matrix from the embeddings

        :param content: The list of sentences
        :param hidden: Which hidden layer to use
        :param reduce_option: The reduce option to run.
        :return: A numpy array matrix of the given content.
        """

        return np.asarray([
            np.squeeze(self.extract_embeddings(t, hidden=hidden, reduce_option=reduce_option).data.cpu().numpy())
            for t in content
        ])

    def __call__(
        self,
        content: List[str],
        hidden: int= -2,
        reduce_option: str = 'mean'
    ) -> ndarray:
        return self.create_matrix(content, hidden, reduce_option)
