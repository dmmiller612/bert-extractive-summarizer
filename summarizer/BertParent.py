from pytorch_pretrained_bert import BertTokenizer, BertModel, GPT2Model, GPT2Tokenizer
import logging
import torch
import numpy as np
from tqdm import tqdm
from numpy import ndarray
from typing import List

logging.basicConfig(level=logging.WARNING)


class BertParent(object):

    def __init__(self, model: str, vector_size: int=None):
        self.model = BertModel.from_pretrained(model)
        self.tokenizer = BertTokenizer.from_pretrained(model)

        if model == 'bert-base-uncased':
            self.vector_size = 768
        elif model == 'bert-large-uncased':
            self.vector_size = 1024
        elif vector_size is None:
            raise RuntimeError("Vector size must be supplied for custom models")
        else:
            self.vector_size = vector_size

        self.model.eval()

    def tokenize_input(self, text: str) -> torch.tensor:
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return torch.tensor([indexed_tokens])

    def extract_embeddings(self, text: str, hidden: int=-2, squeeze: bool=False, reduce_option: str ='mean') -> ndarray:
        tokens_tensor = self.tokenize_input(text)
        hidden_states, pooled = self.model(tokens_tensor)

        if hidden < -1 and hidden > -12:
            if reduce_option == 'max':
                pooled = hidden_states[hidden].max(dim=1)
            elif reduce_option == 'median':
                pooled = hidden_states[hidden].median(dim=1)
            else:
                pooled = hidden_states[hidden].mean(dim=1)

        if squeeze:
            return pooled.detach().numpy().squeeze()

        return pooled

    def create_matrix(self, content: List[str], hidden: int=-2, reduce_option: str = 'mean') -> ndarray:
        train_vec = np.zeros((len(content), self.vector_size))
        for i, t in tqdm(enumerate(content)):
            train_vec[i] = self.extract_embeddings(t, hidden=hidden, reduce_option=reduce_option).data.numpy()
        return train_vec

    def __call__(self, content: List[str], hidden: int=-2, reduce_option: str = 'mean') -> ndarray:
        return self.create_matrix(content, hidden, reduce_option)
