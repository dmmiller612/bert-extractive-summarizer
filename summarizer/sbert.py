from typing import List, Optional, Tuple, Union

import numpy as np

from summarizer.cluster_features import ClusterFeatures
from summarizer.sbert_parent import SBertParent
from summarizer.sentence_handler import SentenceHandler
from summarizer.util import AGGREGATE_MAP


class SBertSummarizer:
    """
    The SBert Summarizer.

    This is based on the Sentence Bert Summarizer.
    """

    def __init__(
        self,
        model: str = 'all-mpnet-base-v2',
        sentence_handler: SentenceHandler = SentenceHandler(),
        random_state: int = 12345
    ):
        """
        SBert Summarizer

        :param model: The model for the sentence transformer.
        :sentence_handler: The handler to process sentences. If want to use coreference, instantiate and pass.
        :param random_state: The random state to reproduce summarizations.
        """
        self.random_state = random_state
        np.random.seed(random_state)
        self.model = SBertParent(model)
        self.sentence_handler = sentence_handler

    def calculate_elbow(
        self,
        body: str,
        algorithm: str = 'kmeans',
        min_length: int = 40,
        max_length: int = 600,
        k_max: int = None,
    ) -> List[float]:
        """
        Calculates elbow across the clusters.

        :param body: The input body to summarize.
        :param algorithm: The algorithm to use for clustering.
        :param min_length: The min length to use.
        :param max_length: The max length to use.
        :param k_max: The maximum number of clusters to search.
        :return: List of elbow inertia values.
        """
        sentences = self.sentence_handler(body, min_length, max_length)

        if k_max is None:
            k_max = len(sentences) - 1

        hidden = self.model(sentences)
        elbow = ClusterFeatures(
            hidden, algorithm, random_state=self.random_state).calculate_elbow(k_max)

        return elbow

    def calculate_optimal_k(
        self,
        body: str,
        algorithm: str = 'kmeans',
        min_length: int = 40,
        max_length: int = 600,
        k_max: int = None,
    ):
        """
        Calculates the optimal Elbow K.

        :param body: The input body to summarize.
        :param algorithm: The algorithm to use for clustering.
        :param min_length: The min length to use.
        :param max_length: The max length to use.
        :param k_max: The maximum number of clusters to search.
        :return:
        """
        sentences = self.sentence_handler(body, min_length, max_length)

        if k_max is None:
            k_max = len(sentences) - 1

        hidden = self.model(sentences)
        optimal_k = ClusterFeatures(
            hidden, algorithm, random_state=self.random_state).calculate_optimal_cluster(k_max)

        return optimal_k

    def cluster_runner(
        self,
        sentences: List[str],
        ratio: float = 0.2,
        algorithm: str = 'kmeans',
        use_first: bool = True,
        num_sentences: int = 3,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Runs the cluster algorithm based on the hidden state. Returns both the embeddings and sentences.

        :param sentences: Content list of sentences.
        :param ratio: The ratio to use for clustering.
        :param algorithm: Type of algorithm to use for clustering.
        :param use_first: Return the first sentence in the output (helpful for news stories, etc).
        :param num_sentences: Number of sentences to use for summarization.
        :return: A tuple of summarized sentences and embeddings
        """
        if num_sentences is not None:
            num_sentences = num_sentences - 1 if use_first else num_sentences

        hidden = self.model(sentences)
        hidden_args = ClusterFeatures(
            hidden, algorithm, random_state=self.random_state).cluster(ratio, num_sentences)

        if use_first:

            if not hidden_args:
                hidden_args.append(0)

            elif hidden_args[0] != 0:
                hidden_args.insert(0, 0)

        sentences = [sentences[j] for j in hidden_args]
        embeddings = np.asarray([hidden[j] for j in hidden_args])

        return sentences, embeddings

    def run_embeddings(
        self,
        body: str,
        ratio: float = 0.2,
        min_length: int = 40,
        max_length: int = 600,
        use_first: bool = True,
        algorithm: str = 'kmeans',
        num_sentences: int = None,
        aggregate: str = None,
    ) -> Optional[np.ndarray]:
        """
        Preprocesses the sentences, runs the clusters to find the centroids, then combines the embeddings.

        :param body: The raw string body to process
        :param ratio: Ratio of sentences to use
        :param min_length: Minimum length of sentence candidates to utilize for the summary.
        :param max_length: Maximum length of sentence candidates to utilize for the summary
        :param use_first: Whether or not to use the first sentence
        :param algorithm: Which clustering algorithm to use. (kmeans, gmm)
        :param num_sentences: Number of sentences to use. Overrides ratio.
        :param aggregate: One of mean, median, max, min. Applied on zero axis
        :return: A summary embedding
        """
        sentences = self.sentence_handler(body, min_length, max_length)

        if sentences:
            _, embeddings = self.cluster_runner(sentences, ratio, algorithm, use_first, num_sentences)

            if aggregate is not None:
                assert aggregate in [
                    'mean', 'median', 'max', 'min'], "aggregate must be mean, min, max, or median"
                embeddings = AGGREGATE_MAP[aggregate](embeddings, axis=0)

            return embeddings

        return None

    def run(
        self,
        body: str,
        ratio: float = 0.2,
        min_length: int = 40,
        max_length: int = 600,
        use_first: bool = True,
        algorithm: str = 'kmeans',
        num_sentences: int = None,
        return_as_list: bool = False,
    ) -> Union[List, str]:
        """
        Preprocesses the sentences, runs the clusters to find the centroids, then combines the sentences.

        :param body: The raw string body to process
        :param ratio: Ratio of sentences to use
        :param min_length: Minimum length of sentence candidates to utilize for the summary.
        :param max_length: Maximum length of sentence candidates to utilize for the summary
        :param use_first: Whether or not to use the first sentence
        :param algorithm: Which clustering algorithm to use. (kmeans, gmm)
        :param num_sentences: Number of sentences to use (overrides ratio).
        :param return_as_list: Whether or not to return sentences as list.
        :return: A summary sentence
        """
        sentences = self.sentence_handler(body, min_length, max_length)

        if sentences:
            sentences, _ = self.cluster_runner(sentences, ratio, algorithm, use_first, num_sentences)

        if return_as_list:
            return sentences
        else:
            return ''.join(sentences)

    def __call__(
        self,
        body: str,
        ratio: float = 0.2,
        min_length: int = 40,
        max_length: int = 600,
        use_first: bool = True,
        algorithm: str = 'kmeans',
        num_sentences: int = None,
        return_as_list: bool = False,
    ) -> str:
        """
        (utility that wraps around the run function)
        Preprocesses the sentences, runs the clusters to find the centroids, then combines the sentences.

        :param body: The raw string body to process.
        :param ratio: Ratio of sentences to use.
        :param min_length: Minimum length of sentence candidates to utilize for the summary.
        :param max_length: Maximum length of sentence candidates to utilize for the summary.
        :param use_first: Whether or not to use the first sentence.
        :param algorithm: Which clustering algorithm to use. (kmeans, gmm)
        :param num_sentences: Number of sentences to use (overrides ratio).
        :param return_as_list: Whether or not to return sentences as list.
        :return: A summary sentence.
        """
        return self.run(body, ratio, min_length, max_length,
                        use_first, algorithm, num_sentences, return_as_list)
