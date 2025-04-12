import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from tqdm import tqdm
import re
import time
import gc

DATA_FOLDER = 'data/'
FN_DF_FILTER = 'df_news_filter.csv'
FN_DF_LABEL = 'df_news_label.csv'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

path_df_filter = DATA_FOLDER + FN_DF_FILTER
df_filter = pd.read_csv(path_df_filter)

def inference(tokenizer, model, prompt, max_new_tokens=1):
    input = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    output = model.generate(
        input.input_ids,
        attention_mask=input.attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=0.6,
        top_p=0.95,
        do_sample=True,
        # repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )
    # decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    input_size = input.input_ids.shape[1]
    idx_first_output_token = input_size
    first_output_token_id = output[:, idx_first_output_token:].squeeze()
    answer = tokenizer.decode(first_output_token_id, skip_special_tokens=False)
    return answer

N_SAMPLE = 30
list_content = df_filter['content'][:N_SAMPLE]
df_filter['content'][49]
list_label = [
    0, 1, 1, 0, 1, 0, 1, 0, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
    1, 1, 1, 0, 0, 1, 0, 0, 0, 0,
    1, 1, 0, 0, 1, 0, 1, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 1, 0, 0, 0
]

N_SIM = 10
list_column = ['model', 'max_new_tokens', 'time', 'queryA', 'queryB']
list_model_path = [
    # 'google/gemma-3-1b-it',
    # 'google/gemma-3-1b-pt',
    # 'google/txgemma-2b-predict',
    # 'google/txgemma-9b-predict',
    # 'google/txgemma-9b-chat',
    # 'Qwen/Qwen2.5-Omni-7B', # transformers version failure
    'Qwen/Qwen2.5-7B-Instruct-1M', # best performance
    'microsoft/Phi-4-mini-instruct' # second best performance
    # 'nvidia/Llama-3.1-Nemotron-Nano-8B-v1', # good 
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", # good
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", # low performance
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", # too low performance
]

df = pd.DataFrame(columns=list_column)
for model_path in tqdm(list_model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    _ = model.to(DEVICE)
    _ = model.config.temperature
    _ = model.config.top_p
    # model.config.pad_token_id

    # list_max_new_tokens = [1, 10, 20]
    list_max_new_tokens = [1]
    for max_new_tokens in tqdm(list_max_new_tokens, leave=False):
        list_prec = []
        start = time.perf_counter()
        for i, content in tqdm(enumerate(list_content), total=len(list_content), leave=False):
            list_prec_temp = []
            list_query = [
                f"Is this article related to train accident? Answer with YES or NO. Answer:",
                f"Is this article related to train crash? Answer with YES or NO. Answer:"
            ]
            for query in list_query:
                prompt = f"context: {content}\n\n query:{query}"
                list_answer = [inference(tokenizer, model, prompt, max_new_tokens=max_new_tokens) for _ in range(N_SIM)]
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
    
    del model
    gc.collect()
    torch.cuda.empty_cache()

df = df.round(3)
path_df_label = DATA_FOLDER + FN_DF_LABEL
df.to_csv(path_df_label, index=False)
