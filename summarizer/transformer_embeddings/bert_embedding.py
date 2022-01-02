from typing import List, Union

import numpy as np
import torch
from numpy import ndarray
from transformers import (AlbertModel, AlbertTokenizer, BertModel,
                          BertTokenizer, DistilBertModel, DistilBertTokenizer,
                          PreTrainedModel, PreTrainedTokenizer, XLMModel,
                          XLMTokenizer, XLNetModel, XLNetTokenizer)


class BertEmbedding:
    """Bert Embedding Handler for BERT models."""

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
        custom_model: PreTrainedModel = None,
        custom_tokenizer: PreTrainedTokenizer = None,
        gpu_id: int = 0,
    ):
        """
        Bert Embedding Constructor. Source for Bert embedding processing.

        :param model: Model is the string path for the bert weights. If given a keyword, the s3 path will be used.
        :param custom_model: This is optional if a custom bert model is used.
        :param custom_tokenizer: Place to use custom tokenizer.
        """
        base_model, base_tokenizer = self.MODELS.get(model, (None, None))

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            assert (
                isinstance(gpu_id, int) and (0 <= gpu_id and gpu_id < torch.cuda.device_count())
            ), f"`gpu_id` must be an integer between 0 to {torch.cuda.device_count() - 1}. But got: {gpu_id}"

            self.device = torch.device(f"cuda:{gpu_id}")

        if custom_model:
            self.model = custom_model.to(self.device)
        else:
            self.model = base_model.from_pretrained(
                model, output_hidden_states=True).to(self.device)

        if custom_tokenizer:
            self.tokenizer = custom_tokenizer
        else:
            self.tokenizer = base_tokenizer.from_pretrained(model)

        self.model.eval()

    def tokenize_input(self, text: str) -> torch.tensor:
        """
        Tokenizes the text input.

        :param text: Text to tokenize.
        :return: Returns a torch tensor.
        """
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return torch.tensor([indexed_tokens]).to(self.device)

    def _pooled_handler(self, hidden: torch.Tensor,
                        reduce_option: str) -> torch.Tensor:
        """
        Handles torch tensor.

        :param hidden: The hidden torch tensor to process.
        :param reduce_option: The reduce option to use, such as mean, etc.
        :return: Returns a torch tensor.
        """

        if reduce_option == 'max':
            return hidden.max(dim=1)[0].squeeze()

        elif reduce_option == 'median':
            return hidden.median(dim=1)[0].squeeze()

        return hidden.mean(dim=1).squeeze()

    def extract_embeddings(
        self,
        text: str,
        hidden: Union[List[int], int] = -2,
        reduce_option: str = 'mean',
        hidden_concat: bool = False,
    ) -> torch.Tensor:
        """
        Extracts the embeddings for the given text.

        :param text: The text to extract embeddings for.
        :param hidden: The hidden layer(s) to use for a readout handler.
        :param reduce_option: How we should reduce the items.
        :param hidden_concat: Whether or not to concat multiple hidden layers.
        :return: A torch vector.
        """
        tokens_tensor = self.tokenize_input(text)
        pooled, hidden_states = self.model(tokens_tensor)[-2:]

        # deprecated temporary keyword functions.
        if reduce_option == 'concat_last_4':
            last_4 = [hidden_states[i] for i in (-1, -2, -3, -4)]
            cat_hidden_states = torch.cat(tuple(last_4), dim=-1)
            return torch.mean(cat_hidden_states, dim=1).squeeze()

        elif reduce_option == 'reduce_last_4':
            last_4 = [hidden_states[i] for i in (-1, -2, -3, -4)]
            return torch.cat(tuple(last_4), dim=1).mean(axis=1).squeeze()

        elif type(hidden) == int:
            hidden_s = hidden_states[hidden]
            return self._pooled_handler(hidden_s, reduce_option)

        elif hidden_concat:
            last_states = [hidden_states[i] for i in hidden]
            cat_hidden_states = torch.cat(tuple(last_states), dim=-1)
            return torch.mean(cat_hidden_states, dim=1).squeeze()

        last_states = [hidden_states[i] for i in hidden]
        hidden_s = torch.cat(tuple(last_states), dim=1)

        return self._pooled_handler(hidden_s, reduce_option)

    def create_matrix(
        self,
        content: List[str],
        hidden: Union[List[int], int] = -2,
        reduce_option: str = 'mean',
        hidden_concat: bool = False,
    ) -> ndarray:
        """
        Create matrix from the embeddings.

        :param content: The list of sentences.
        :param hidden: Which hidden layer to use.
        :param reduce_option: The reduce option to run.
        :param hidden_concat: Whether or not to concat multiple hidden layers.
        :return: A numpy array matrix of the given content.
        """

        return np.asarray([
            np.squeeze(self.extract_embeddings(
                t, hidden=hidden, reduce_option=reduce_option, hidden_concat=hidden_concat
            ).data.cpu().numpy()) for t in content
        ])

    def __call__(
        self,
        content: List[str],
        hidden: Union[List[int], int] = -2,
        reduce_option: str = 'mean',
        hidden_concat: bool = False,
    ) -> ndarray:
        """
        Create matrix from the embeddings.

        :param content: The list of sentences.
        :param hidden: Which hidden layer to use.
        :param reduce_option: The reduce option to run.
        :param hidden_concat: Whether or not to concat multiple hidden layers.
        :return: A numpy array matrix of the given content.
        """
        return self.create_matrix(content, hidden, reduce_option, hidden_concat)
