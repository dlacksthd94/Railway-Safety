#!/usr/bin/bash

set -e

source ~/anaconda3/etc/profile.d/conda.sh

conda deactivate
conda env remove -n railway -y
conda create -n railway python=3.9 -y

conda activate railway

pip cache purge
conda clean --all -y

pip install simcse
conda install -c conda-forge spacy -y
# conda install -c conda-forge cupy -y
# python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf
conda install -c conda-forge pandas -y
conda install -c conda-forge selenium -y
wget https://storage.googleapis.com/chrome-for-testing-public/115.0.5763.0/linux64/chrome-linux64.zip
unzip -j chrome-linux64.zip
wget https://storage.googleapis.com/chrome-for-testing-public/115.0.5763.0/linux64/chromedriver-linux64.zip
unzip -j chromedriver-linux64.zip
pip install newspaper3k
pip install lxml_html_clean
# https://raw.githubusercontent.com/GoogleChromeLabs/chrome-for-testing/refs/heads/main/data/known-good-versions-with-downloads.json

# pip install webdriver_manager
# pip install newspaper3k
# pip install lxml_html_clean

# pip cache purge
# conda clean --all -y