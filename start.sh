#!/bin/bash

set -e

export NLTK_DATA="./nltk_data"
mkdir -p $NLTK_DATA
python3 -m nltk.downloader -d $NLTK_DATA punkt stopwords

exec uvicorn main:app --host 0.0.0.0 --port 10000

