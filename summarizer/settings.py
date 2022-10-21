import os
import logging

APP_ENV = os.environ.get("APP_ENV", "local")
APP_VERSION = os.environ.get("APP_VERSION", "Version 0.11.0")

PORT = int(os.environ.get("PORT", 8080))
HOST = os.environ.get("HOST", "0.0.0.0")
FLASK_DEBUG = False  # Do not use debug mode in production
FLASK_LOG_LEVEL = os.environ.get("FLASK_LOG_LEVEL", logging.INFO)

NUM_SENTENCES = int(os.environ.get("NUM_SENTENCES", 5))
OUTPUT_RATIO = float(os.environ.get("OUTPUT_RATIO", 0.0))
MIN_INPUT_LENGTH = int(os.environ.get("MIN_INPUT_LENGTH", 10))
MAX_INPUT_LENGTH = int(os.environ.get("MAX_INPUT_LENGTH", 512))
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "distilbert-base-uncased")

HIDDEN = int(os.environ.get("HIDDEN", -2))
REDUCE = os.environ.get("REDUCE", "mean")
