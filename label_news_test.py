import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import time
import gc
from transformers import pipeline

DATA_FOLDER = 'data/'
FN_DF_SAMPLE = 'df_news_label_sample.csv'
FN_DF_LABEL_TEST = 'df_news_label_test.csv'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

path_df_label = DATA_FOLDER + FN_DF_SAMPLE
df_label = pd.read_csv(path_df_label)

N_SIM = 5
list_column = ['model', 'time', 'queryA', 'queryB', 'queryC']
dict_models = {
    # 'google/gemma-3-1b-it',
    # 'google/gemma-3-1b-pt': {},
    # 'google/txgemma-2b-predict',
    # 'google/txgemma-9b-predict',
    # 'google/txgemma-9b-chat': {},
    # 'Qwen/Qwen2.5-Omni-7B': {}, # transformers version failure
    'Qwen/Qwen2.5-7B-Instruct-1M': {}, # third best performance
    'microsoft/Phi-4-mini-instruct': {}, # best performance
    'nvidia/Llama-3.1-Nemotron-Nano-8B-v1': {}, # good but sometimes bad
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {}, # second best performance
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {}, # low performance
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {}, # too low performance
    "facebook/bart-large-mnli": {'candidate_labels': ["YES", "NO"]} # worst performance
}

df = pd.DataFrame(columns=list_column)
for model_path, config in tqdm(dict_models.items()):
    pipe = pipeline(model=model_path, device_map=DEVICE, **config)
    task = pipe.task
    if task == 'text-generation' and pipe.generation_config.pad_token_id is None:
        pipe.generation_config.pad_token_id = pipe.tokenizer.eos_token_id
    if config.get('temperature', None) == 0 or config.get('do_sample', None) == False:
        n_sim = 1
    else:
        n_sim = N_SIM

    answer_constraint = 'Answer only YES or NO.'
    if task == 'zero-shot-classification':
        answer_constraint = ''

    list_prec = []
    start = time.perf_counter()
    for i, row in tqdm(df_label.iterrows(), total=df_label.shape[0], leave=False):
        _, _, content, label = row
        list_prec_by_query = []
        list_query = [
            f"Is this article's main topic reporting a train accident? {answer_constraint}", # best
            f"Is the primary focus of this article to report a train accident? {answer_constraint}", # good but sometimes bad
            f"Does the article report on a train accident as its main topic? {answer_constraint}", # not bad
        ]
        
        for query in list_query:
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
            prec = (np.array(list_answer_binary) == (label == 'YES')).mean()
            list_prec_by_query.append(prec)
        list_prec.append(list_prec_by_query)
    end = time.perf_counter()
    elapsed_time = end - start

    prec = np.stack(list_prec)
    prec_mean = prec.mean(axis=0).tolist()
    row = [model_path, elapsed_time] + prec_mean
    df.loc[len(df)] = row

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

df = df.round(3)
path_df_label_test = DATA_FOLDER + FN_DF_LABEL_TEST
df.to_csv(path_df_label_test, index=False)
