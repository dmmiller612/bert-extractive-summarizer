from flask import Flask
from flask import request, jsonify, abort, make_response
from flask_cors import CORS, cross_origin
from flask.logging import default_handler
from werkzeug.middleware.proxy_fix import ProxyFix
from nltk import tokenize
from typing import List
from summarizer import Summarizer, settings

app = Flask(__name__)
app.logger.addHandler(default_handler)
app.logger.setLevel(settings.FLASK_LOG_LEVEL)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1, x_proto=1)
CORS(app)


summarizer = Summarizer(
    model=settings.DEFAULT_MODEL,
    hidden=int(settings.HIDDEN),
    reduce_option=settings.REDUCE
)

app.logger.info("Confirming environment ...")
app.logger.info(settings.APP_ENV)
app.logger.info(settings.APP_VERSION)

app.logger.info(f"Using Model: {settings.DEFAULT_MODEL}")


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


@app.route('/status')
@cross_origin()
def status():
    return 'ok'


@app.route('/')
@cross_origin()
def root():
    return 'ok'


@app.route('/summarize', methods=['POST'])
@cross_origin()
def summarize_text():
    ratio = float(request.args.get('ratio', settings.OUTPUT_RATIO))
    num_sentences = int(request.args.get('num_sentences', settings.NUM_SENTENCES))
    min_length = int(request.args.get('min_length', settings.MIN_INPUT_LENGTH))
    max_length = int(request.args.get('max_length', settings.MAX_INPUT_LENGTH))

    data = request.data
    if not data:
        abort(make_response(jsonify(message="Request must be plain text"), 400))

    parsed = Parser(data).convert_to_paragraphs()
    if 0 < ratio < 100:
        summary = summarizer(parsed, ratio=ratio, min_length=min_length, max_length=max_length)
    else:
        summary = summarizer(parsed, num_sentences=num_sentences, min_length=min_length, max_length=max_length)

    return jsonify({
        'summary': summary
    })


if __name__ == '__main__':
    app.run(host=settings.HOST, port=int(settings.PORT), debug=settings.FLASK_DEBUG)
