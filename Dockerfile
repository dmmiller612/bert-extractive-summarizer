# Version 0.11.0
FROM python:3.9.13-slim-bullseye

RUN apt-get update && \
    apt-get install -y sudo \
    build-essential \
    curl \
    libcurl4-openssl-dev \
    libssl-dev \
    wget \
    libxrender-dev \
    libxext6 \
    libsm6 \
    openssl

RUN mkdir -p /opt/service
COPY requirements.txt /opt/service
COPY summarizer /opt/service/summarizer
COPY tests /opt/service/tests
COPY application.py /opt/service
COPY wsgi.py /opt/service
COPY gunicorn_config.py /opt/service
WORKDIR /opt/service

RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt && \
    python3 -m spacy download en_core_web_sm && \
    python3 -m nltk.downloader punkt && \
    pytest -v

ENTRYPOINT ["gunicorn", "--config", "gunicorn_config.py", "wsgi:app", "--no-sendfile"]
