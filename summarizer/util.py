from typing import List, Tuple

import numpy as np

from summarizer.cluster_features import ClusterFeatures

AGGREGATE_MAP = {
    'mean': np.mean,
    'min': np.min,
    'median': np.median,
    'max': np.max,
}


def run_cluster(
    sentences: List[str],
    hidden: np.ndarray,
    random_state: int,
    ratio: float = 0.2,
    algorithm: str = 'kmeans',
    use_first: bool = True,
    num_sentences: int = 3,
) -> Tuple[List[str], np.ndarray]:
    """
    Runs the cluster algorithm based on the hidden state. Returns both the embeddings and sentences.

    :param sentences: Content list of sentences.
    :param hidden: The hidden representations of the summarizer model.
    :param random_state: The random state for numpy, torch, sklearn.
    :param ratio: The ratio to use for clustering.
    :param algorithm: Type of algorithm to use for clustering.
    :param use_first: Return the first sentence in the output (helpful for news stories, etc).
    :param num_sentences: Number of sentences to use for summarization.
    :return: A tuple of summarized sentences and embeddings
    """
    first_embedding = None

    if use_first:
        num_sentences = num_sentences - 1 if num_sentences else num_sentences

        if len(sentences) <= 1 or (num_sentences and num_sentences <= 1):
            return sentences, hidden

        first_embedding = hidden[0, :]
        hidden = hidden[1:, :]

    summary_sentence_indices = ClusterFeatures(
        hidden, algorithm, random_state=random_state).cluster(ratio, num_sentences)

    if use_first:
        if summary_sentence_indices:
            # adjust for the first sentence to the right.
            summary_sentence_indices = [i + 1 for i in summary_sentence_indices]
            summary_sentence_indices.insert(0, 0)
        else:
            summary_sentence_indices.append(0)

        hidden = np.vstack([first_embedding, hidden])

    sentences = [sentences[j] for j in summary_sentence_indices]
    embeddings = np.asarray([hidden[j] for j in summary_sentence_indices])

    return sentences, embeddings


def calculate_elbow():
    pass



