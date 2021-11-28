from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class SBertParent:
    """
    SBert Parent. This is for the SentenceTransformer Package.
    """

    def __init__(
        self,
        model: str
    ):
        """
        SBert Parent Handler

        :param model: The model string for SentenceTransformer
        """
        self.sbert_model = SentenceTransformer(model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sbert_model.to(self.device)

    def extract_embeddings(
        self,
        sentences: List[str]
    ) -> np.ndarray:
        """
        Calculates sentence embeddings.

        :param sentences: The sentences to summarizer.
        :return Numpy array of sentences.
        """
        embeddings = self.sbert_model.encode(sentences)
        return embeddings

    def __call__(
        self,
        sentences: List[str]
    ) -> np.ndarray:
        """
        Calculates sentence embeddings.

        :param sentences: The sentences to summarizer.
        :return Numpy array of sentences.
        """
        return self.extract_embeddings(sentences)
