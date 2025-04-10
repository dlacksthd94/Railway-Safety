#!/usr/bin/bash

ENV_NAME="rw_safety"

set -e
source ~/miniconda3/etc/profile.d/conda.sh

##### rw_scraping #####
conda deactivate
conda env remove -n $ENV_NAME -y
conda create -n $ENV_NAME python=3.10 -y

conda activate $ENV_NAME

pip cache purge
conda clean --all -y

# conda install -c conda-forge spacy -y
# # conda install -c conda-forge cupy -y
# # python -m spacy download en_core_web_sm
# python -m spacy download en_core_web_trf
pip install pandas
pip install selenium

# LATEST VERSION HERE!!! https://googlechromelabs.github.io/chrome-for-testing/
rm -rf chrome-linux64* chromedriver-linux64*
wget https://storage.googleapis.com/chrome-for-testing-public/135.0.7049.42/linux64/chrome-linux64.zip
unzip chrome-linux64.zip
wget https://storage.googleapis.com/chrome-for-testing-public/135.0.7049.42/linux64/chromedriver-linux64.zip
unzip chromedriver-linux64.zip
pip install newspaper3k
pip install lxml_html_clean
# https://raw.githubusercontent.com/GoogleChromeLabs/chrome-for-testing/refs/heads/main/data/known-good-versions-with-downloads.json
pip install feedparser
pip install requests
pip install timeout-decorator
pip install undetected-chromedriver
pip install streamlit
pip install transformers
pip install mistral-common
pip install mistral-inference