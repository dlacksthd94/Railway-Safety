import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import os
from transformers import pipeline

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
dict_models = {
    # 'google/gemma-3-1b-it',
    # 'google/gemma-3-1b-pt': {},
    # 'google/txgemma-2b-predict',
    # 'google/txgemma-9b-predict',
    # 'google/txgemma-9b-chat': {}, # no authorization
    # 'Qwen/Qwen2.5-Omni-7B': {}, # transformers version failure
    'Qwen/Qwen2.5-7B-Instruct-1M': {}, # third best performance
    'microsoft/Phi-4-mini-instruct': {'truncation': True, 'use_cache': False}, # best performance
    'nvidia/Llama-3.1-Nemotron-Nano-8B-v1': {}, # good but sometimes bad
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {}, # second best performance
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {}, # low performance
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {}, # too low performance
    "facebook/bart-large-mnli": {'candidate_labels': ["YES", "NO"]} # worst performance
}

MODEL_PATH = 'microsoft/Phi-4-mini-instruct'
config = dict_models[MODEL_PATH]
pipe = pipeline(model=MODEL_PATH, device_map=DEVICE, **config)
task = pipe.task
if task == 'text-generation':
    if pipe.generation_config.pad_token_id is None:
        pipe.generation_config.pad_token_id = pipe.tokenizer.eos_token_id
    if pipe.generation_config.temperature == 0 or pipe.generation_config.do_sample == False:
        n_sim = 1
    else:
        n_sim = N_SIM
    answer_constraint = 'Answer only YES or NO.'
elif task == 'zero-shot-classification':
    answer_constraint = ''
else:
    raise ValueError(f"Unsupported task: {task}")

for idx, row in tqdm(df_label.iterrows(), total=len(df_label), desc='Labeling'):
    if row[COLUMNS_LABEL].notna().all():
        continue
    series_content = row[COLUMNS_CONTENT]
    for column_name, content in series_content.items():
        query = f"Is this article's main topic reporting a train accident? {answer_constraint}" # best
        prompt = f"Context: {content}\n\nQuestion: {query}\n\nAnswer:"
        list_answer = []
        if task == 'zero-shot-classification':
            output = pipe(prompt)
            answer = output['labels'][np.argmax(output['scores'])]
            list_answer.append(answer)
        elif task == 'text-generation':
            for _ in range(n_sim):
                output = pipe(prompt, max_new_tokens=1)
                answer = output[0]['generated_text'].split('Answer:')[-1].strip()
                list_answer.append(answer)
        list_answer_binary = [1 if re.search(r'yes', answer, re.IGNORECASE) else 0 for answer in list_answer]
        is_yes = np.array(list_answer_binary).mean()
        df_label.loc[idx, 'label_' + column_name] = is_yes.item()

    if idx % 10 == 0:
        df_label.to_csv(path_df_label, index=False)

df_label.to_csv(path_df_label, index=False)