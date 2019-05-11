# Bert Extractive Summarizer

This repo is the generalization of the lecture-summarizer repo. This tool utilizes the HuggingFace Pytorch BERT library 
to run extractive summarizations. This works by first embedding the sentences, then running a clustering algorithm, finding 
the sentences that are closest to the cluster's centroids.

## How to Use

#### Simple Example
```python
from summarizer.model_processors import SingleModelProcessor

body = 'text you want to summarize'
summary = SingleModelProcessor()(body)

```

