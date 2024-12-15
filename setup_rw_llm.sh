#!/usr/bin/bash

set -e
source ~/anaconda3/etc/profile.d/conda.sh

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

# CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8 --port 3000