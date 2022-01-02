# Bert Extractive Summarizer

![Build Status](https://github.com/dmmiller612/bert-extractive-summarizer/actions/workflows/test.yml/badge.svg)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/dmmiller612/bert-extractive-summarizer)
<img src="https://img.shields.io/pypi/v/bert-extractive-summarizer.svg" />

This repo is the generalization of the lecture-summarizer repo. This tool utilizes the HuggingFace Pytorch transformers library
to run extractive summarizations. This works by first embedding the sentences, then running a clustering algorithm, finding 
the sentences that are closest to the cluster's centroids. This library also uses coreference techniques, utilizing the 
https://github.com/huggingface/neuralcoref library to resolve words in summaries that need more context. The greedyness of 
the neuralcoref library can be tweaked in the CoreferenceHandler class.

As of the most recent version of bert-extractive-summarizer, by default, CUDA is used if a gpu is available.

Paper: https://arxiv.org/abs/1906.04165

### Try the Online Demo:

[Distill Bert Summarization Demo](https://smrzr.io)

# Table of Contents
1. [Install](#install)
2. [Examples](#examples)
   1. [Simple Example](#simple-example)
   2. [SBert](#use-sbert)
   3. [Retrieve Embeddings](#retrieve-embeddings)
   4. [Use Coreference](#use-coreference)
   5. [Custom Model Example](#custom-model-example)
   6. [Large Example](#large-example)
3. [Calculating Elbow](#calculating-elbow)
4. [Running the Service](#running-the-service)

## Install

```bash
pip install bert-extractive-summarizer
```

## Examples

### Simple Example
```python
from summarizer import Summarizer

body = 'Text body that you want to summarize with BERT'
body2 = 'Something else you want to summarize with BERT'
model = Summarizer()
model(body)
model(body2)
```

#### Specifying number of sentences

Number of sentences can be supplied as a ratio or an integer. Examples are provided below.

```python
from summarizer import Summarizer
body = 'Text body that you want to summarize with BERT'
model = Summarizer()
result = model(body, ratio=0.2)  # Specified with ratio
result = model(body, num_sentences=3)  # Will return 3 sentences 
```

#### Using multiple hidden layers as the embedding output

You can also concat the summarizer embeddings for clustering. A simple example is below.

```python
from summarizer import Summarizer
body = 'Text body that you want to summarize with BERT'
model = Summarizer('distilbert-base-uncased', hidden=[-1,-2], hidden_concat=True)
result = model(body, num_sentences=3)
```

### Use SBert
One can use Sentence Bert with bert-extractive-summarizer with the newest version. It is based off the paper here:
https://arxiv.org/abs/1908.10084, and the library here: https://www.sbert.net/. To get started,
first install SBERT:

```
pip install -U sentence-transformers
```

Then a simple example is the following:

```python
from summarizer.sbert import SBertSummarizer

body = 'Text body that you want to summarize with BERT'
model = SBertSummarizer('paraphrase-MiniLM-L6-v2')
result = model(body, num_sentences=3)
```

It is worth noting that all the features that you can do with the main Summarizer class, you can also do with SBert.

### Retrieve Embeddings
You can also retrieve the embeddings of the summarization. Examples are below:

```python
from summarizer import Summarizer
body = 'Text body that you want to summarize with BERT'
model = Summarizer()
result = model.run_embeddings(body, ratio=0.2)  # Specified with ratio. 
result = model.run_embeddings(body, num_sentences=3)  # Will return (3, N) embedding numpy matrix.
result = model.run_embeddings(body, num_sentences=3, aggregate='mean')  # Will return Mean aggregate over embeddings. 
```

### Use Coreference
First ensure you have installed neuralcoref and spacy. It is worth noting that neuralcoref does not work with spacy > 0.2.1.
```bash
pip install spacy
pip install transformers # > 4.0.0
pip install neuralcoref

python -m spacy download en_core_web_md
```

Then to to use coreference, run the following:

```python
from summarizer import Summarizer
from summarizer.text_processors.coreference_handler import CoreferenceHandler

handler = CoreferenceHandler(greedyness=.4)
# How coreference works:
# >>>handler.process('''My sister has a dog. She loves him.''', min_length=2)
# ['My sister has a dog.', 'My sister loves a dog.']

body = 'Text body that you want to summarize with BERT'
body2 = 'Something else you want to summarize with BERT'
model = Summarizer(sentence_handler=handler)
model(body)
model(body2)
```

### Custom Model Example
```python
from transformers import *

# Load model, model config and tokenizer via Transformers
custom_config = AutoConfig.from_pretrained('allenai/scibert_scivocab_uncased')
custom_config.output_hidden_states=True
custom_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
custom_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', config=custom_config)

from summarizer import Summarizer

body = 'Text body that you want to summarize with BERT'
body2 = 'Something else you want to summarize with BERT'
model = Summarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer)
model(body)
model(body2)
```

### Large Example

```python
from summarizer import Summarizer

body = '''
The Chrysler Building, the famous art deco New York skyscraper, will be sold for a small fraction of its previous sales price.
The deal, first reported by The Real Deal, was for $150 million, according to a source familiar with the deal.
Mubadala, an Abu Dhabi investment fund, purchased 90% of the building for $800 million in 2008.
Real estate firm Tishman Speyer had owned the other 10%.
The buyer is RFR Holding, a New York real estate company.
Officials with Tishman and RFR did not immediately respond to a request for comments.
It's unclear when the deal will close.
The building sold fairly quickly after being publicly placed on the market only two months ago.
The sale was handled by CBRE Group.
The incentive to sell the building at such a huge loss was due to the soaring rent the owners pay to Cooper Union, a New York college, for the land under the building.
The rent is rising from $7.75 million last year to $32.5 million this year to $41 million in 2028.
Meantime, rents in the building itself are not rising nearly that fast.
While the building is an iconic landmark in the New York skyline, it is competing against newer office towers with large floor-to-ceiling windows and all the modern amenities.
Still the building is among the best known in the city, even to people who have never been to New York.
It is famous for its triangle-shaped, vaulted windows worked into the stylized crown, along with its distinctive eagle gargoyles near the top.
It has been featured prominently in many films, including Men in Black 3, Spider-Man, Armageddon, Two Weeks Notice and Independence Day.
The previous sale took place just before the 2008 financial meltdown led to a plunge in real estate prices.
Still there have been a number of high profile skyscrapers purchased for top dollar in recent years, including the Waldorf Astoria hotel, which Chinese firm Anbang Insurance purchased in 2016 for nearly $2 billion, and the Willis Tower in Chicago, which was formerly known as Sears Tower, once the world's tallest.
Blackstone Group (BX) bought it for $1.3 billion 2015.
The Chrysler Building was the headquarters of the American automaker until 1953, but it was named for and owned by Chrysler chief Walter Chrysler, not the company itself.
Walter Chrysler had set out to build the tallest building in the world, a competition at that time with another Manhattan skyscraper under construction at 40 Wall Street at the south end of Manhattan. He kept secret the plans for the spire that would grace the top of the building, building it inside the structure and out of view of the public until 40 Wall Street was complete.
Once the competitor could rise no higher, the spire of the Chrysler building was raised into view, giving it the title.
'''

model = Summarizer()
result = model(body, min_length=60)
full = ''.join(result)
print(full)
"""
The Chrysler Building, the famous art deco New York skyscraper, will be sold for a small fraction of its previous sales price. 
The building sold fairly quickly after being publicly placed on the market only two months ago.
The incentive to sell the building at such a huge loss was due to the soaring rent the owners pay to Cooper Union, a New York college, for the land under the building.'
Still the building is among the best known in the city, even to people who have never been to New York.
"""
```

### Calculating Elbow

As of bert-extractive-summarizer version 0.7.1, you can also calculate ELBOW to determine the optimal cluster. Below 
shows a sample example in how to retrieve the list of inertias.

```python
from summarizer import Summarizer

body = 'Your Text here.'
model = Summarizer()
res = model.calculate_elbow(body, k_max=10)
print(res)
```

You can also find the optimal number of sentences with elbow using the following algorithm.

```python
from summarizer import Summarizer

body = 'Your Text here.'
model = Summarizer()
res = model.calculate_optimal_k(body, k_max=10)
print(res)
```

## Summarizer Options

```
model = Summarizer(
    model: This gets used by the hugging face bert library to load the model, you can supply a custom trained model here
    custom_model: If you have a pre-trained model, you can add the model class here.
    custom_tokenizer:  If you have a custom tokenizer, you can add the tokenizer here.
    hidden: Needs to be negative, but allows you to pick which layer you want the embeddings to come from.
    reduce_option: It can be 'mean', 'median', or 'max'. This reduces the embedding layer for pooling.
    sentence_handler: The handler to process sentences. If want to use coreference, instantiate and pass CoreferenceHandler instance
)

model(
    body: str # The string body that you want to summarize
    ratio: float # The ratio of sentences that you want for the final summary
    min_length: int # Parameter to specify to remove sentences that are less than 40 characters
    max_length: int # Parameter to specify to remove sentences greater than the max length,
    num_sentences: Number of sentences to use. Overrides ratio if supplied.
)
```

## Running the Service

There is a provided flask service and corresponding Dockerfile. Running the service is simple, and can be done though 
the Makefile with the two commands:

```
make docker-service-build
make docker-service-run
```

This will use the Bert-base-uncased model, which has a small representation. The docker run also accepts a variety of 
arguments for custom and different models. This can be done through a command such as:

```
docker build -t summary-service -f Dockerfile.service ./
docker run --rm -it -p 5000:5000 summary-service:latest -model bert-large-uncased
```

Other arguments can also be passed to the server. Below includes the list of available arguments.

* -greediness: Float parameter that determines how greedy nueralcoref should be
* -reduce: Determines the reduction statistic of the encoding layer (mean, median, max).
* -hidden: Determines the hidden layer to use for embeddings (default is -2)
* -port: Determines the port to use.
* -host: Determines the host to use.

Once the service is running, you can make a summarization command at the `http://localhost:5000/summarize` endpoint. 
This endpoint accepts a text/plain input which represents the text that you want to summarize. Parameters can also be 
passed as request arguments. The accepted arguments are:

* ratio: Ratio of sentences to summarize to from the original body. (default to 0.2)
* min_length: The minimum length to accept as a sentence. (default to 25)
* max_length: The maximum length to accept as a sentence. (default to 500)

An example of a request is the following:

```
POST http://localhost:5000/summarize?ratio=0.1

Content-type: text/plain

Body:
The Chrysler Building, the famous art deco New York skyscraper, will be sold for a small fraction of its previous sales price.
The deal, first reported by The Real Deal, was for $150 million, according to a source familiar with the deal.
Mubadala, an Abu Dhabi investment fund, purchased 90% of the building for $800 million in 2008.
Real estate firm Tishman Speyer had owned the other 10%.
The buyer is RFR Holding, a New York real estate company.
Officials with Tishman and RFR did not immediately respond to a request for comments.
It's unclear when the deal will close.
The building sold fairly quickly after being publicly placed on the market only two months ago.
The sale was handled by CBRE Group.
The incentive to sell the building at such a huge loss was due to the soaring rent the owners pay to Cooper Union, a New York college, for the land under the building.
The rent is rising from $7.75 million last year to $32.5 million this year to $41 million in 2028.
Meantime, rents in the building itself are not rising nearly that fast.
While the building is an iconic landmark in the New York skyline, it is competing against newer office towers with large floor-to-ceiling windows and all the modern amenities.
Still the building is among the best known in the city, even to people who have never been to New York.
It is famous for its triangle-shaped, vaulted windows worked into the stylized crown, along with its distinctive eagle gargoyles near the top.
It has been featured prominently in many films, including Men in Black 3, Spider-Man, Armageddon, Two Weeks Notice and Independence Day.
The previous sale took place just before the 2008 financial meltdown led to a plunge in real estate prices.
Still there have been a number of high profile skyscrapers purchased for top dollar in recent years, including the Waldorf Astoria hotel, which Chinese firm Anbang Insurance purchased in 2016 for nearly $2 billion, and the Willis Tower in Chicago, which was formerly known as Sears Tower, once the world's tallest.
Blackstone Group (BX) bought it for $1.3 billion 2015.
The Chrysler Building was the headquarters of the American automaker until 1953, but it was named for and owned by Chrysler chief Walter Chrysler, not the company itself.
Walter Chrysler had set out to build the tallest building in the world, a competition at that time with another Manhattan skyscraper under construction at 40 Wall Street at the south end of Manhattan. He kept secret the plans for the spire that would grace the top of the building, building it inside the structure and out of view of the public until 40 Wall Street was complete.
Once the competitor could rise no higher, the spire of the Chrysler building was raised into view, giving it the title.

Response:

{
    "summary": "The Chrysler Building, the famous art deco New York skyscraper, will be sold for a small fraction of its previous sales price. The buyer is RFR Holding, a New York real estate company. The incentive to sell the building at such a huge loss was due to the soaring rent the owners pay to Cooper Union, a New York college, for the land under the building."
}
```
