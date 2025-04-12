import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from tqdm import tqdm
import re
import time
import gc
import os

DATA_FOLDER = 'data/'
FN_DF_FILTER = 'df_news_filter.csv'
FN_DF_LABEL = 'df_news_label.csv'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

path_df_label = DATA_FOLDER + FN_DF_LABEL
if os.path.exists(path_df_label):
    df_label = pd.read_csv(path_df_label)
else:
    path_df_filter = DATA_FOLDER + FN_DF_FILTER
    df_label = pd.read_csv(path_df_filter)
    df_label['yes'] = None

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

N_SIM = 10
# MODEL_PATH = 'Qwen/Qwen2.5-7B-Instruct-1M' # best performance
MODEL_PATH = 'microsoft/Phi-4-mini-instruct' # second best performance

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
_ = model.to(DEVICE)
_ = model.config.temperature
_ = model.config.top_p
# model.config.pad_token_id

max_new_tokens = 1

for i, row in tqdm(df_label.iterrows(), total=len(df_label), desc='Labeling'):
    if df_label.loc[i, 'yes'] is not None:
        continue
    content = row['content']
    query = f"Is this article related to train accident? Answer with YES or NO. Answer:"
    prompt = f"context: {content}\n\n query:{query}"
    list_answer = [inference(tokenizer, model, prompt, max_new_tokens=max_new_tokens) for _ in range(N_SIM)]
    list_answer_binary = [1 if re.search(r'yes', answer, re.IGNORECASE) else 0 for answer in list_answer]
    yes = np.array(list_answer_binary).mean()
    df_label.loc[i, 'yes'] = yes

    if i % 10 == 0:
        path_df_label = DATA_FOLDER + FN_DF_LABEL
        df_label.to_csv(path_df_label, index=False)
