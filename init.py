import nltk
nltk.download('punkt')
from nltk import tokenize
from summarizer import Summarizer, TransformerSummarizer

if __name__ == '__main__':
    print(f"Using Model: bert-base-uncased")

    summarizer = Summarizer(
        model='bert-base-uncased'
    )
