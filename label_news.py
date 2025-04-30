import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from tqdm import tqdm
import re
import time
import os
import utils_inference

DATA_FOLDER = 'data/'
FN_DF_FILTER = 'df_news_filter.csv'
FN_DF_LABEL = 'df_news_label.csv'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
COLUMNS_CONTENT = ['np_url', 'tf_url', 'rd_url', 'gs_url', 'np_html', 'tf_html', 'rd_html', 'gs_html']
COLUMNS_LABEL = ['label_np_url', 'label_tf_url', 'label_rd_url', 'label_gs_url', 'label_np_html', 'label_tf_html', 'label_rd_html', 'label_gs_html']

path_df_label = DATA_FOLDER + FN_DF_LABEL
path_df_filter = DATA_FOLDER + FN_DF_FILTER

if os.path.exists(path_df_label):
    df_label = pd.read_csv(path_df_label)
else:
    df_filter = pd.read_csv(path_df_filter)
    df_label = df_filter.copy()
    df_label[COLUMNS_LABEL] = float('nan')

N_SIM = 1
MODEL_PATH = 'microsoft/Phi-4-mini-instruct'
dict_model = {
    # 'Qwen/Qwen2.5-7B-Instruct-1M': {
    #     'max_new_tokens': 1,
    #     'temperature': 0.7,
    #     'top_p': 0.8,
    #     'do_sample': True,
    #     # 'do_sample': False,
    #     'repetition_penalty': 1.05,
    # }, # good
    'microsoft/Phi-4-mini-instruct': {
        'max_new_tokens': 1,
        'do_sample': False, # no need to set 'temperature' to 0
    }, # best
    # 'nvidia/Llama-3.1-Nemotron-Nano-8B-v1': {
    #     'max_new_tokens': 1,
    #     'temperature': 0.6,
    #     'top_p': 0.95,
    #     'do_sample': True,
    #     # 'do_sample': False,
    # }, # good but highly dependes on prompt
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
    #     'max_new_tokens': 1,
    #     'temperature': 0.6,
    #     'top_p': 0.95,
    #     'do_sample': True,
    #     # 'do_sample': False,
    # }, # good
}

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
_ = model.to(DEVICE)

config = dict_model[MODEL_PATH]
config['use_cache'] = False

for idx, row in tqdm(df_label.iterrows(), total=len(df_label), desc='Labeling'):
    if row[COLUMNS_LABEL].notna().all():
        continue
    series_content = row[COLUMNS_CONTENT]
    for column_name, content in series_content.items():
        query = "Is this article's main topic reporting a train accident? Answer only YES or NO."
        prompt = f"context: {content}\n\nquery:{query}\n\nanswer:"
        list_answer = [utils_inference.inference(tokenizer, model, prompt, config, DEVICE) for _ in range(N_SIM)]
        list_answer_binary = [1 if re.search(r'yes', answer, re.IGNORECASE) else 0 for answer in list_answer]
        yes = np.array(list_answer_binary).mean()
        df_label.loc[idx, 'label_' + column_name] = yes.item()

    if idx % 10 == 0:
        df_label.to_csv(path_df_label, index=False)
df_label.to_csv(path_df_label, index=False)