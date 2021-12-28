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
