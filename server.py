from flask import Flask
from flask import request, jsonify, abort, make_response
from flask_cors import CORS
import nltk
nltk.download('punkt')
from nltk import tokenize
from typing import List
import argparse
from summarizer import Summarizer, TransformerSummarizer


app = Flask(__name__)
CORS(app)


class Parser(object):

    def __init__(self, raw_text: bytes):
        self.all_data = str(raw_text, 'utf-8').split('\n')

    def __isint(self, v) -> bool:
        try:
            int(v)
            return True
        except:
            return False

    def __should_skip(self, v) -> bool:
        return self.__isint(v) or v == '\n' or '-->' in v

    def __process_sentences(self, v) -> List[str]:
        sentence = tokenize.sent_tokenize(v)
        return sentence

    def save_data(self, save_path, sentences) -> None:
        with open(save_path, 'w') as f:
            for sentence in sentences:
                f.write("%s\n" % sentence)

    def run(self) -> List[str]:
        total: str = ''
        for data in self.all_data:
            if not self.__should_skip(data):
                cleaned = data.replace('&gt;', '').replace('\n', '').strip()
                if cleaned:
                    total += ' ' + cleaned
        sentences = self.__process_sentences(total)
        return sentences

    def convert_to_paragraphs(self) -> str:
        sentences: List[str] = self.run()
        return ' '.join([sentence.strip() for sentence in sentences]).strip()


@app.route('/summarize', methods=['POST'])
def convert_raw_text():
    ratio = float(request.args.get('ratio', 0.2))
    min_length = int(request.args.get('min_length', 25))
    max_length = int(request.args.get('max_length', 500))

    data = request.data
    if not data:
        abort(make_response(jsonify(message="Request must have raw text"), 400))

    parsed = Parser(data).convert_to_paragraphs()
    summary = summarizer(parsed, ratio=ratio, min_length=min_length, max_length=max_length)

    return jsonify({
        'summary': summary
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-model', dest='model', default='bert-base-uncased', help='The model to use')
    parser.add_argument('-transformer-type',
                        dest='transformer_type', default=None,
                        help='Huggingface transformer class key')
    parser.add_argument('-transformer-key', dest='transformer_key', default=None,
                        help='The transformer key for huggingface. For example bert-base-uncased for Bert Class')
    parser.add_argument('-greediness', dest='greediness', help='', default=0.45)
    parser.add_argument('-reduce', dest='reduce', help='', default='mean')
    parser.add_argument('-hidden', dest='hidden', help='', default=-2)
    parser.add_argument('-port', dest='port', help='', default=5000)
    parser.add_argument('-host', dest='host', help='', default='0.0.0.0')

    args = parser.parse_args()

    if args.transformer_type is not None:
        print(f"Using Model: {args.transformer_type}")
        assert args.transformer_key is not None, 'Transformer Key cannot be none with the transformer type'

        summarizer = TransformerSummarizer(
            transformer_type=args.transformer_type,
            transformer_model_key=args.transformer_key,
            hidden=int(args.hidden),
            reduce_option=args.reduce
        )

    else:
        print(f"Using Model: {args.model}")

        summarizer = Summarizer(
            model=args.model,
            hidden=int(args.hidden),
            reduce_option=args.reduce
        )

    app.run(host=args.host, port=int(args.port))
