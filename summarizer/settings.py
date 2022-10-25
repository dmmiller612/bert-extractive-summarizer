import os
import logging

APP_ENV = os.environ.get("APP_ENV", "local")
APP_VERSION = os.environ.get("APP_VERSION", "Version 0.12.0")

PORT = int(os.environ.get("PORT", 8080))
HOST = os.environ.get("HOST", "0.0.0.0")
FLASK_DEBUG = False  # Do not use debug mode in production
FLASK_LOG_LEVEL = os.environ.get("FLASK_LOG_LEVEL", logging.INFO)
CACHE_DIR = os.environ.get('CACHE_DIR', '/tmp/cache')
CACHE_TIMEOUT = int(os.environ.get('CACHE_TIMEOUT', 1800))
RATE_LIMIT = os.environ.get("RATE_LIMIT", "10 per second").split(",")

NUM_SENTENCES = int(os.environ.get("NUM_SENTENCES", 5))
OUTPUT_RATIO = float(os.environ.get("OUTPUT_RATIO", 0.0))
MIN_INPUT_LENGTH = int(os.environ.get("MIN_INPUT_LENGTH", 10))
MAX_INPUT_LENGTH = int(os.environ.get("MAX_INPUT_LENGTH", 512))
USE_FIRST_SENTENCE = bool(os.environ.get("USE_FIRST_SENTENCE", True))
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "distilbert-base-uncased")
DEFAULT_SBERT_MODEL = os.environ.get("DEFAULT_SBERT_MODEL", "paraphrase-MiniLM-L6-v2")
DEFAULT_ENGINE = os.environ.get("DEFAULT_ENGINE", "sbert")

HIDDEN = int(os.environ.get("HIDDEN", -2))
REDUCE = os.environ.get("REDUCE", "mean")
