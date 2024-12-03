#!/usr/bin/bash

set -e

##### rw_scraping #####
source ~/anaconda3/etc/profile.d/conda.sh

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

wget https://storage.googleapis.com/chrome-for-testing-public/115.0.5763.0/linux64/chrome-linux64.zip
unzip chrome-linux64.zip
wget https://storage.googleapis.com/chrome-for-testing-public/115.0.5763.0/linux64/chromedriver-linux64.zip
unzip chromedriver-linux64.zip
pip install newspaper3k
pip install lxml_html_clean
# https://raw.githubusercontent.com/GoogleChromeLabs/chrome-for-testing/refs/heads/main/data/known-good-versions-with-downloads.json

##### rw_llm #####
conda deactivate
conda env remove -n rw_llm -y
conda create -n rw_llm python=3.11 -y

conda activate rw_llm

pip cache purge
conda clean --all -y

pip install pymupdf4llm
pip install sglang

pip install aiohttp
pip install orjson
pip install uvicorn
pip install uvloop
pip install fastapi
pip install huggingface_hub
pip install transformers
pip install torch==2.4
pip install psutil
pip install zmq
pip install interegular

pip install outlines==0.0.46
# pip install openai
# pip install transformers datasets accelerate torch
# pip install llama-cpp-python
# pip install exllamav2 transformers torch
# pip install mamba_ssm transformers torch
pip install vllm==0.5.5

pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
pip install python-multipart

pi install sentence_transformers

# CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 --port 3300