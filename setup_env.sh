#!/usr/bin/bash

ENV_NAME="rw_safety"

set -e
source ~/miniconda3/etc/profile.d/conda.sh

##### rw_scraping #####
if conda info --envs | grep -q "^$ENV_NAME\s"; then
    conda deactivate
    conda env remove -n $ENV_NAME -y
fi
conda create -n $ENV_NAME python=3.10 -y

conda activate $ENV_NAME

pip cache purge
conda clean --all -y

pip install pandas
pip install feedparser
pip install selenium
pip install requests
pip install newspaper3k lxml_html_clean trafilatura readability-lxml goose3
pip install streamlit
pip install transformers

find . -name "chrome-*" -exec rm -rf {} +
find . -name "chromedriver-*" -exec rm -rf {} +
# LATEST VERSION HERE!!! https://googlechromelabs.github.io/chrome-for-testing/
VERSION="135.0.7049.95"
OS="$(uname -s)"
ARCH="$(uname -m)"
case "${OS}" in
  Linux)
    PLATFORM="linux64"
    DOWNLOADER="wget -q --show-progress -O"
    ;;
  Darwin)
    if [[ "${ARCH}" = "arm64" ]]; then
      PLATFORM="mac-arm64"
    else
      PLATFORM="mac-x64"
    fi
    DOWNLOADER="curl --fail --location --progress-bar -O"
    ;;
  *)
    echo "❌ Unsupported OS: ${OS}" >&2
    exit 1
    ;;
esac
URL="https://storage.googleapis.com/chrome-for-testing-public/${VERSION}/${PLATFORM}/chrome-${PLATFORM}.zip"
eval "${DOWNLOADER} ${URL}"
URL="https://storage.googleapis.com/chrome-for-testing-public/${VERSION}/${PLATFORM}/chromedriver-${PLATFORM}.zip"
eval "${DOWNLOADER} ${URL}"
unzip "chrome-${PLATFORM}.zip"
unzip "chromedriver-${PLATFORM}.zip"
