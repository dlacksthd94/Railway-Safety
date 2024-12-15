#!/usr/bin/bash

set -e
source ~/anaconda3/etc/profile.d/conda.sh

##### rw_scraping #####
conda deactivate
conda env remove -n rw_scraping -y
conda create -n rw_scraping python=3.9 -y

conda activate rw_scraping

pip cache purge
conda clean --all -y

pip install simcse
conda install -c conda-forge spacy -y
# conda install -c conda-forge cupy -y
# python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf
conda install -c conda-forge pandas -y
conda install -c conda-forge selenium -y

# LATEST VERSION HERE!!! https://googlechromelabs.github.io/chrome-for-testing/
rm -rf chrome-linux64* chromedriver-linux64*
# wget https://storage.googleapis.com/chrome-for-testing-public/115.0.5763.0/linux64/chrome-linux64.zip
wget https://storage.googleapis.com/chrome-for-testing-public/131.0.6778.108/linux64/chrome-linux64.zip
unzip chrome-linux64.zip
# wget https://storage.googleapis.com/chrome-for-testing-public/115.0.5763.0/linux64/chromedriver-linux64.zip
wget https://storage.googleapis.com/chrome-for-testing-public/131.0.6778.108/linux64/chromedriver-linux64.zip
unzip chromedriver-linux64.zip
pip install newspaper3k
pip install lxml_html_clean
# https://raw.githubusercontent.com/GoogleChromeLabs/chrome-for-testing/refs/heads/main/data/known-good-versions-with-downloads.json