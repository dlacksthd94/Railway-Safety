from transformers import pipeline
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from pprint import pprint
import numpy as np
from tqdm import tqdm
import re
import time

DATA_FOLDER = 'data/'
FN_DF_FILTER = 'df_news_filter.csv'
FN_DF_LABEL = 'df_news_label.csv'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

path_df_filter = DATA_FOLDER + FN_DF_FILTER
df_filter = pd.read_csv(path_df_filter)

def inference(tokenizer, model, prompts, max_new_tokens=1):
    inputs = tokenizer(prompts, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=0.6,
        top_p=0.95,
        do_sample=True,
        # repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )
    # decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    input_size = inputs.input_ids.shape[1]
    idx_first_output_token = input_size
    first_output_token_ids = outputs[:, idx_first_output_token:]
    answers = [tokenizer.decode(first_output_token_id, skip_special_tokens=False) for first_output_token_id in first_output_token_ids]
    return answers

N_SAMPLE = 10
list_content = df_filter['content'][:N_SAMPLE]
# list_label = ['NO', 'YES', 'YES', 'NO', 'YES', 'NO', 'YES', 'NO', 'YES', 'YES']
list_label = [0, 1, 1, 0, 1, 0, 1, 0, 1, 1]

N_SIM = 50
list_column = ['model', 'max_new_tokens', 'time', 'wo_reason', 'w_reason']
list_model_path = [
    # 'google/gemma-3-1b-it',
    # 'google/gemma-3-1b-pt',
    # 'google/txgemma-2b-predict',
    # 'google/txgemma-9b-predict',
    # 'google/txgemma-9b-chat',
    # 'Qwen/Qwen2.5-Omni-7B', # transformers version failure
    'Qwen/Qwen2.5-7B-Instruct-1M',
    'nvidia/Llama-3.1-Nemotron-Nano-8B-v1',
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", # too low performance
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    'microsoft/Phi-4-mini-instruct'
]

df = pd.DataFrame(columns=list_column)
for model_path in tqdm(list_model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    _ = model.to(DEVICE)
    _ = model.config.temperature
    _ = model.config.top_p
    # model.config.pad_token_id

    for max_new_tokens in tqdm([1, 2], leave=False):
        list_prec = []
        start = time.perf_counter()
        for i, content in tqdm(enumerate(list_content), total=len(list_content), leave=False):
            list_prec_temp = []
            list_query = [
                f"Is this article related to train accident? Answer with YES or NO. Answer:",
                f"Is this article related to train accident? Answer with YES or NO. Then, explain the reason. Answer:"
            ]
            for query in list_query:
                prompt = f"context: {content}\n\n query:{query}"
                prompts = [prompt] * N_SIM
                list_answer = inference(tokenizer, model, prompts, max_new_tokens=max_new_tokens)
                list_answer_binary = [1 if re.search(r'yes', answer, re.IGNORECASE) else 0 for answer in list_answer]
                prec_temp = (np.array(list_answer_binary) == list_label[i]).mean()
                list_prec_temp.append(prec_temp)
            list_prec.append(list_prec_temp)
        end = time.perf_counter()
        elapsed_time = end - start

        prec = np.stack(list_prec)
        prec_mean = prec.mean(axis=0).tolist()
        row = [model_path, max_new_tokens, elapsed_time] + prec_mean
        df.loc[len(df)] = row
    
    torch.cuda.empty_cache()
df = df.round(2)
path_df_label = DATA_FOLDER + FN_DF_LABEL
df.to_csv(path_df_label, index=False)
