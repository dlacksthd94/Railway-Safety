import pandas as pd
import json
from transformers import pipeline
import torch
from pprint import pprint
from tqdm import tqdm
import os
from itertools import chain
import re

dict_target_info = {
    'railroad': {
        1: 'Railroad Name',
        2: 'Other Railroad Name',
        3: 'Maintenance Railroad Name',
    },
    'time': {
        5: 'Date',
        "5a": 'Day of Week',
        6: 'Time',
    },
    'location': {
        7: 'Nearest Station',
        9: 'County Name',
        10: 'State Name',
        11: 'City Name',
        12: 'Highway Name',
    },
    'highway user': {
        13: 'Highway User',
        14: 'Estimated Vehicle Speed',
        15: 'Vehicle Direction',
        16: 'Highway User Position',
        38: 'User Age',
        39: 'User Sex',
        40: 'User Struck By Second Train',
        41: 'Highway User Action',
    },
    'train': {
        # 17: 'Equipment Involved',
        # 18: 'Railroad Car Unit Position',
        19: 'Equipment Struck',
        24: 'Equipment Type',
        26: "Track Name",
        30: 'Train Speed',
        31: 'Train Direction',
    },
    'hazmat': {
        '20a': 'Hazmat Involvement',
        '20b': 'Hazmat Released by',
        '20c': 'Hazmat Released Quantity',
        '20c': 'Hazmat Released Quantity',
    },
    'environment': {
        21: 'Temperature',
        22: 'Visibility',
        23: 'Weather Condition',
        34: 'Roadway Condition',
    },
    # 'warning': {
        # 32: [f'Crossing Warning Expanded {str(i)}' for i in range(1, 13)],
        # 35: 'Crossing Warning Location',
        # 36: 'Warning Connected To Signal',
        # 37: 'Crossing Illuminated',
    # },
    'damage': {
        44: 'Driver Condition',
        45: 'Driver In Vehicle',
        '46a': 'Crossing Users Killed',
        '46b': 'Crossing Users Injured',
        48: 'Number Vehicle Occupants',
        '49a': 'Employees Killed',
        '49b': 'Employees Injured',
        50: 'Number People On Train',
        '52a': 'Passengers Killed',
        '52b': 'Passengers Injured',
    },
    # 'analysis': {
    #     42: 'Driver Passed Vehicle',
    #     43: 'View Obstruction',
    #     54: 'Narrative',
    # },
}

COLUMNS_CONTENT = ['np_url', 'tf_url', 'rd_url', 'gs_url', 'np_html', 'tf_html', 'rd_html', 'gs_html']
COLUMNS_LABEL = ['label_np_url', 'label_tf_url', 'label_rd_url', 'label_gs_url', 'label_np_html', 'label_tf_html', 'label_rd_html', 'label_gs_html']
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def extract_keywords(path_form57_json, path_form57_json_group, path_df_form57_retrieval, path_df_news_label, config_retrieval):
    _, model, n_generate, question_batch = config_retrieval.to_tuple()

    with open(path_form57_json, 'r') as f:
        dict_form57 = json.load(f)
    
    if question_batch == 'group':
        with open(path_form57_json_group, 'r') as f:
            dict_form57_group = json.load(f)

    list_col = list(dict_form57)

    SCRAPE_VERSION = 'rd_url'
    if os.path.exists(path_df_form57_retrieval):
        df_retrieval = pd.read_csv(path_df_form57_retrieval)
        df_retrieval[list_col] = df_retrieval[list_col].fillna('')
    else:
        df_label = pd.read_csv(path_df_news_label)
        df_label = df_label[(df_label[COLUMNS_LABEL] == 1).any(axis=1)]
        sr_content = df_label[SCRAPE_VERSION]
        mask = df_label['label_' + SCRAPE_VERSION] == 1
        df_retrieval = df_label.drop(COLUMNS_CONTENT + COLUMNS_LABEL, axis=1)
        df_retrieval[SCRAPE_VERSION] = sr_content
        df_retrieval = df_retrieval[mask]
        df_retrieval.loc[:, list_col] = ''

    dict_model = {
        'microsoft/Phi-4-mini-instruct': {
            'max_new_tokens': 512,
            'truncation': True,
            'use_cache': True,
        },
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
            'max_new_tokens': 512,
            'truncation': True,
            'use_cache': True,
            'do_sample': False,
        },
        'Qwen/Qwen2.5-7B-Instruct-1M': {
            'max_new_tokens': 512,
            'truncation': True,
            'use_cache': True,
        },
        'nvidia/Llama-3.1-Nemotron-Nano-8B-v1': {
            'max_new_tokens': 512,
            'truncation': True,
            'use_cache': True,
        },
    }

    config = dict_model[model]
    pipe = pipeline(model=model, device_map=DEVICE, **config)
    if pipe.generation_config.pad_token_id is None:
        pipe.generation_config.pad_token_id = pipe.tokenizer.eos_token_id
    if pipe.generation_config.temperature == 0 or pipe.generation_config.do_sample == False:
        n_sim = 1
    else:
        n_sim = n_generate

    ############ extract information using csv-version json

    question_base = (
            "From the context, answer the following question: "
        )
    answer_format = (
        "The answer should be short and without explanation.\n"
        "If options are provided, answer only with codes without labels.\n"
        "If you cannot find the relevant information, answer with 'Unknown'."
    )

    for idx_content, row in tqdm(df_retrieval.iterrows(), total=df_retrieval.shape[0]):
        if row.iat[-1]:
            continue
        content = row[SCRAPE_VERSION]
        
        if question_batch == 'single':
            for entry_idx, entry in tqdm(dict_form57.items(), leave=False):
                name = entry.get('name', '')
                options = str(entry.get('choices', ''))
                
                question = question_base + '\n' + f'({entry_idx}): ' + name + f'(options: {options})' + '\n' + answer_format
                prompt = f"Context:\n{content}\n\nQuestion:\n{question}\n\nAnswer:\n"
                
                output = pipe(prompt)#, max_new_tokens=30)
                answer = output[0]['generated_text'].split('Answer:\n')[-1]

                df_retrieval.loc[idx_content, name] = answer
            # df_retrieval.loc[idx_content, :].to_dict()

        elif question_batch == 'group':
            for group_name, group in tqdm(dict_form57_group.items(), leave=False):
                question = question_base + '\n'
                for entry_idx in tqdm(group, leave=False):
                    entry = dict_form57[entry_idx]
                    name = entry.get('name', '')
                    options = str(entry.get('choices', ''))
                    question += f'({entry_idx}): ' + name + f'(options: {options})' + '\n'
                question += answer_format
                prompt = f"Context:\n{content}\n\nQuestion:\n{question}\n\nAnswer:\n({group[0]}):"
                
                output = pipe(prompt)#, max_new_tokens=30)
                answer = output[0]['generated_text'].split('Answer:\n')[-1]

                dict_answer = dict(re.findall(r'\((.+)\):\s*([^\n]+)', answer))
                # pprint(dict_answer)

                for entry_idx in group:
                    df_retrieval.loc[idx_content, entry_idx] = dict_answer.get(entry_idx, '')
            # df_retrieval.loc[idx_content, :].to_dict()

        if idx_content % 10 == 0:
            df_retrieval.to_csv(path_df_form57_retrieval, index=False)

    df_retrieval.to_csv(path_df_form57_retrieval, index=False)
    return df_retrieval