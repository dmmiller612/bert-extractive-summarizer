from flask import Flask
from flask import request, jsonify, abort, make_response
from flask_cors import CORS, cross_origin
from flask.logging import default_handler
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_limiter import Limiter, RateLimitExceeded
from flask_caching import Cache
from nltk import tokenize
from typing import List
import hashlib
from summarizer import Summarizer, settings
from summarizer.sbert import SBertSummarizer
from summarizer.text_processors.coreference_handler import CoreferenceHandler


def make_cache_key():
    """Hash the request args and content"""
    args = request.args
    hs = hashlib.md5()
    # update hash with args, data and url
    args_key = "".join([str(v) for k, v in args.items()]) + str(request.data) + request.url
    hs.update(args_key.encode('utf-8'))
    cache_key = hs.hexdigest()
    return cache_key


def get_fwd_remote_address():
    """
    Try to get real client IP from 'X-Forwarded-For' headers sent by nginx
    (renamed as 'HTTP_X_FORWARDED_FOR' by werkzeug/flask)
    If not, fall back to whatever the load balancer gives us
    :return: actual client ip address, not address of proxy or load balancer
    """
    fwd_ip = request.environ.get('HTTP_X_FORWARDED_FOR', '').split(',')[0]
    client_ip_address = fwd_ip or request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
    return client_ip_address


# Set up simple file cache
cache = Cache(config={'CACHE_TYPE': 'SimpleCache',
                      'CACHE_DIR': settings.CACHE_DIR,
                      'CACHE_DEFAULT_TIMEOUT': settings.CACHE_TIMEOUT})

# Set up basic rate limiting
limiter = Limiter(
    key_func=get_fwd_remote_address,
    default_limits=settings.RATE_LIMIT
)

limiter.request_filter(lambda: request.method.upper() == 'OPTIONS')
limiter.request_filter(lambda: str(request.url_rule).startswith('/static/'))

app = Flask(__name__)
app.logger.addHandler(default_handler)
app.logger.setLevel(settings.FLASK_LOG_LEVEL)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1, x_proto=1)
CORS(app)

cache.init_app(app)
limiter.init_app(app)

handler = CoreferenceHandler() if settings.USE_COREFERENCE else None
summarizer = Summarizer(
    model=settings.DEFAULT_MODEL,
    hidden=int(settings.HIDDEN),
    reduce_option=settings.REDUCE,
    sentence_handler=handler
)

sbert_summarizer = SBertSummarizer(
    model=settings.DEFAULT_SBERT_MODEL,
    sentence_handler=handler
)

app.logger.info("Confirming environment ...")
app.logger.info(settings.APP_ENV)
app.logger.info(settings.APP_VERSION)

app.logger.info(f"Using Coreference Engine: {settings.USE_COREFERENCE}")
app.logger.info(f"Using Model: {settings.DEFAULT_MODEL}")
app.logger.info(f"Using SBert Model: {settings.DEFAULT_SBERT_MODEL}")


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
@limiter.exempt
@cross_origin()
def status():
    return 'ok'


@app.route('/')
@limiter.exempt
@cross_origin()
def root():
    return 'ok'


@app.route('/summarize', methods=['POST'])
@cross_origin()
@cache.cached(key_prefix=make_cache_key)
def summarize_text():
    engine = request.args.get('engine', settings.DEFAULT_ENGINE)
    ratio = float(request.args.get('ratio', settings.OUTPUT_RATIO))
    num_sentences = int(request.args.get('num_sentences', settings.NUM_SENTENCES))
    min_length = int(request.args.get('min_length', settings.MIN_INPUT_LENGTH))
    max_length = int(request.args.get('max_length', settings.MAX_INPUT_LENGTH))
    use_first = request.args.get('use_first', settings.USE_FIRST_SENTENCE) in {'True', 'true', 1, True}

    engines = {"bert": summarizer, "sbert": sbert_summarizer}
    summary_engine = engines.get(engine, sbert_summarizer)

    data = request.data
    if not data:
        abort(make_response(jsonify(message="Request must be plain text"), 400))

    parsed = Parser(data).convert_to_paragraphs()
    if 0 < ratio < 100:
        summary = summary_engine(parsed,
                                 ratio=ratio,
                                 min_length=min_length,
                                 max_length=max_length,
                                 use_first=use_first)
    else:
        summary = summary_engine(parsed,
                                 num_sentences=num_sentences,
                                 min_length=min_length,
                                 max_length=max_length,
                                 use_first=use_first)

    return jsonify({
        'summary': summary
    })


@app.errorhandler(RateLimitExceeded)
def ratelimit_handler(e):
    message = "Rate limit {} exceeded."
    return make_response(
        jsonify(error=message.format(e.description))
        , 429
    )


if __name__ == '__main__':
    app.run(host=settings.HOST, port=int(settings.PORT), debug=settings.FLASK_DEBUG)
