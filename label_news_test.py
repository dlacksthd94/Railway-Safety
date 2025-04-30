import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from tqdm import tqdm
import re
import time
import gc

DATA_FOLDER = 'data/'
FN_DF_SAMPLE = 'df_news_label_sample.csv'
FN_DF_LABEL_TEST = 'df_news_label_test.csv'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

path_df_label = DATA_FOLDER + FN_DF_SAMPLE
df_label = pd.read_csv(path_df_label)

def inference(tokenizer, model, prompt, config):
    input = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    output = model.generate(
        input.input_ids,
        attention_mask=input.attention_mask,
        **config
    )
    # decoded_outputs = [tokenizer.decode(o, skip_special_tokens=True) for o in output]
    input_size = input.input_ids.shape[1]
    idx_first_output_token = input_size
    first_output_token_id = output[:, idx_first_output_token:].squeeze()
    answer = tokenizer.decode(first_output_token_id, skip_special_tokens=False)
    return answer

N_SIM = 5
list_column = ['model', 'time', 'queryA', 'queryB', 'queryC']
dict_model = {
    # 'google/gemma-3-1b-it',
    # 'google/gemma-3-1b-pt',
    # 'google/txgemma-2b-predict',
    # 'google/txgemma-9b-predict',
    # 'google/txgemma-9b-chat',
    # 'Qwen/Qwen2.5-Omni-7B', # transformers version failure
    'Qwen/Qwen2.5-7B-Instruct-1M': {
        'max_new_tokens': 1,
        'temperature': 0.7,
        'top_p': 0.8,
        'do_sample': True,
        # 'do_sample': False,
        'repetition_penalty': 1.05,
    }, # best performance
    'microsoft/Phi-4-mini-instruct': {
        'max_new_tokens': 1,
        'temperature': 0.0,
        'do_sample': False,
    }, # second best performance
    'nvidia/Llama-3.1-Nemotron-Nano-8B-v1': {
        'max_new_tokens': 1,
        'temperature': 0.6,
        'top_p': 0.95,
        'do_sample': True,
        # 'do_sample': False,
    }, # good 
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
        'max_new_tokens': 1,
        'temperature': 0.6,
        'top_p': 0.95,
        'do_sample': True,
        # 'do_sample': False,
    }, # good
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", # low performance
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", # too low performance
}

df = pd.DataFrame(columns=list_column)
for model_path, config in tqdm(dict_model.items()):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    _ = model.to(DEVICE)

    list_prec = []
    start = time.perf_counter()
    for i, row in tqdm(df_label.iterrows(), total=df_label.shape[0], leave=False):
        _, _, content, label = row
        list_prec_temp = []
        list_query = [
            # "Is this article reporting a train accident? Answer with YES or NO.",
            # "Is this article's main topic a train accident? Answer with YES or NO.",
            "Is this article's main topic reporting a train accident? Answer only YES or NO.", # best
            "Is the primary focus of this article to report a train accident? Answer only YES or NO.", # good but sometimes bad
            "Does the article report on a train accident as its main topic? Answer only YES or NO.", # not bad
        ]
        for query in list_query:
            prompt = f"context: {content}\n\nquery:{query}\n\nanswer:"
            if model.config.pad_token_id is None:
                config['pad_token_id'] = tokenizer.eos_token_id
            list_answer = [inference(tokenizer, model, prompt, config) for _ in range(N_SIM)]
            list_answer_binary = [1 if re.search(r'yes', answer, re.IGNORECASE) else 0 for answer in list_answer]
            prec_temp = (np.array(list_answer_binary) == (label == 'YES')).mean()
            list_prec_temp.append(prec_temp)
        list_prec.append(list_prec_temp)
    end = time.perf_counter()
    elapsed_time = end - start

    prec = np.stack(list_prec)
    prec_mean = prec.mean(axis=0).tolist()
    row = [model_path, elapsed_time] + prec_mean
    df.loc[len(df)] = row

    del model
    gc.collect()
    torch.cuda.empty_cache()

df = df.round(3)
path_df_label_test = DATA_FOLDER + FN_DF_LABEL_TEST
df.to_csv(path_df_label_test, index=False)
