#!/usr/bin/bash

#################### create miniconda env
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

#################### install packages
pip cache purge
conda clean --all -y

pip install pandas
pip install feedparser requests # for Google RSS
pip install selenium newspaper3k lxml_html_clean trafilatura readability-lxml goose3 # for scraping news articles
pip install streamlit
pip install openai pytesseract # for transcribing report form in json format
pip install transformers accelerate # for labeling news & populating the report form
pip install sentencepiece # for running VLMs from Hugging Face

############ Modify code in `def postprocess` in `class ImageTextToTextPipeline` in transformers/pipelines/image_text_to_text.py
# After `decoded_inputs = self.processor.post_process_image_text_to_text(input_ids, **postprocess_kwargs)` line, add the following codes:
# from itertools import cycle
# decoded_inputs = cycle(decoded_inputs)
# input_texts = cycle(input_texts)

#################### install chrome driver for news scraping
find . -name "chrome-*" -exec rm -rf {} +
find . -name "chromedriver-*" -exec rm -rf {} +
# LATEST VERSION HERE!!! https://googlechromelabs.github.io/chrome-for-testing/
VERSION="136.0.7103.94"
OS="$(uname -s)"
ARCH="$(uname -m)"
case "${OS}" in
  Linux)
    PLATFORM="linux64"
    DOWNLOADER="wget -q --show-progress"
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

# #################### install Google Cloud CLI (optional)
# find . -name "google-cloud*" -exec rm -rf {} +
# # install			https://cloud.google.com/sdk/docs/install
# # authentification	https://cloud.google.com/docs/authentication/set-up-adc-local-dev-environment
# wget https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
# tar -xf google-cloud-cli-linux-x86_64.tar.gz
# ./google-cloud-sdk/install.sh
# ./google-cloud-sdk/bin/gcloud init
# pip install google-cloud-documentai google-cloud-vision

# #################### install AWS Textract (optional)
# find . -name "aws*" -exec rm -rf {} +
# refer to:			https://docs.aws.amazon.com/textract/latest/dg/analyzing-document-text.html?utm_source=chatgpt.com
# wget "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -O awscliv2.zip
# unzip awscliv2.zip
# AWS_DIR="$PWD/aws"
# mkdir "$AWS_DIR/bin"
# ./aws/install -i $AWS_DIR -b "$AWS_DIR/bin"
# if grep -Fxq "export PATH=\"$AWS_DIR/bin:\$PATH\"" ~/.bashrc; then
#   echo "✅ Line already present in ~/.bashrc"
# else
#   echo "export PATH=\"$AWS_DIR/bin:\$PATH\"" >> ~/.bashrc
# fi
# pip install boto3 pdf2image
# aws configure --profile textract

# #################### isntall Azure Form Recognizer (optional)
# https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/concept/choose-model-feature?view=doc-intel-4.0.0#pretrained-document-analysis-models
# pip install azure-ai-formrecognizer
# pip install azure-identity
# pip install azure-ai-documentintelligence
