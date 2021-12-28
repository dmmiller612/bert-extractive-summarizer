from summarizer.sbert_parent import SBertParent
from summarizer.sentence_handler import SentenceHandler
from summarizer.summarize_parent import SummarizeParent


class SBertSummarizer(SummarizeParent):
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
        model_func = SBertParent(model)
        super().__init__(
            model=model_func, sentence_handler=sentence_handler, random_state=random_state
        )

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
