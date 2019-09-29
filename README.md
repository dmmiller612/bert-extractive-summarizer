# Bert Extractive Summarizer

This repo is the generalization of the lecture-summarizer repo. This tool utilizes the HuggingFace Pytorch transformers library
to run extractive summarizations. This works by first embedding the sentences, then running a clustering algorithm, finding 
the sentences that are closest to the cluster's centroids. This library also uses coreference techniques, utilizing the 
https://github.com/huggingface/neuralcoref library to resolve words in summaries that need more context. The greedyness of 
the neuralcoref library can be tweaked in the SingleModel class.

Paper: https://arxiv.org/abs/1906.04165

## Install

#### NOTE: You will need spacy 2.1.3 installed. There is currently an issue with Spacy 2.1.4 that produces segmentation faults. 

With that in mind, the setup.py should install 2.1.3 by default.
```bash
pip install spacy==2.1.3
pip install transformers
```

## How to Use

#### Simple Example
```python
from summarizer import Summarizer

body = 'Text body that you want to summarize with BERT'
body2 = 'Something else you want to summarize with BERT'
model = Summarizer()
model(body)
model(body2)
```

#### Large Example

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
The incentive to sell the building at such a huge loss was due to the soaring rent the owners pay to Cooper Union, a New York college, for the land under the building.Still the building is among the best known in the city, even to people who have never been to New York.'
Still the building is among the best known in the city, even to people who have never been to New York.
"""
```

## Summarizer Options

```
model = Summarizer(
    model: str #This gets used by the hugging face bert library to load the model, you can supply a custom trained model here
    hidden: int # Needs to be negative, but allows you to pick which layer you want the embeddings to come from.
    reduce_option: str # It can be 'mean', 'median', or 'max'. This reduces the embedding layer for pooling.
    greedyness: float # number between 0 and 1. It is used for the coreference model. Anywhere from 0.35 to 0.45 seems to work well.
)

model(
    body: str # The string body that you want to summarize
    ratio: float # The ratio of sentences that you want for the final summary
    min_length: int # Parameter to specify to remove sentences that are less than 40 characters
    max_length: int # Parameter to specify to remove sentences greater than the max length
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


