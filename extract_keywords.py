import pandas as pd
import json
from transformers import pipeline
import torch
import gc
from pprint import pprint
from tqdm import tqdm
import os
from itertools import chain
import re
from PIL import Image
import json5
from openai import OpenAI
from utils import parse_json_from_output

def to_answer_places(dict_form57):
    dict_answer_places = {}
    dict_idx_ap = {}
    for entry_idx, entry in dict_form57.items():
        entry_name = entry['name']
        answer_places = entry['answer_places']
        dict_idx_ap[entry_idx] = []
        if len(answer_places) == 1:
            answer_place_name = next(iter(answer_places.keys()))
            answer_place_info = next(iter(answer_places.values()))
            dict_answer_places[entry_idx] = {'answer_place_name': answer_place_name, 'answer_place_info': answer_place_info}
            dict_idx_ap[entry_idx].append(entry_idx)
        elif len(answer_places) > 1:
            for suffix, (answer_place_name, answer_place_info) in enumerate(answer_places.items(), start=1):
                entry_idx_suffix = entry_idx + '_' + str(suffix)
                dict_answer_places[entry_idx_suffix] = {'answer_place_name': answer_place_name, 'answer_place_info': answer_place_info}
                dict_idx_ap[entry_idx].append(entry_idx_suffix)
        else:
            assert False, 'why # of answer places < 1?'
    return dict_answer_places, dict_idx_ap

def extract_keywords(path_form57_json, path_form57_json_group, path_df_form57_retrieval, path_df_news_articles_filter, path_df_match, path_dict_answer_places, path_form57_img, config):
    _, _, _, json_source = config.conversion.to_tuple()
    _, model, n_generate, question_batch = config.retrieval.to_tuple()

    if json_source == 'None':
        list_answer_places = ["1", "2", "3", "4", "5_month", "5_day", "5_year", "6", "6_ampm", "7", "8", "9", "10", "11", "12", "12_ownership", 
                              "13", "14", "15", "16", "17", "18", "19", "20a", "20b", "20c", 
                              "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "30_record", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", 
                              "44", "45", "46_killed", "46_injured", "47", "48", "49_killed", "49_injured", "50", "51", "52_killed", "52_injured", "53a", "53b", "54", "55", "56", "57"]
    else:
        with open(path_form57_json, 'r') as f:
            dict_form57 = json.load(f)
        dict_answer_places, dict_idx_ap = to_answer_places(dict_form57)
        list_answer_places = list(dict_answer_places.keys())

        with open(path_dict_answer_places, 'w') as f:
            json.dump(dict_answer_places, f, indent=4)
    
    if question_batch == 'group':
        with open(path_form57_json_group, 'r') as f:
            dict_form57_group = json.load(f)

    if os.path.exists(path_df_form57_retrieval):
        df_retrieval = pd.read_csv(path_df_form57_retrieval)
        df_retrieval = df_retrieval.fillna('')
    elif os.path.exists(path_df_match):
        df_match = pd.read_csv(path_df_match)
        df_match = df_match[df_match['match'] == 1]
        df_retrieval = df_match[['news_id', 'url', 'pub_date', 'title', 'content']].copy(deep=True)
        df_retrieval[list_answer_places] = ''
    else:
        df_news_articles_filter = pd.read_csv(path_df_news_articles_filter)
        df_retrieval = df_news_articles_filter.copy(deep=True)
        df_retrieval[list_answer_places] = ''

    config = {
        'max_new_tokens': 512,
        'truncation': True,
        'use_cache': True,
        'do_sample': False,
    }
    
    if question_batch in ['single', 'group']:
        pipe = pipeline(model=model, device_map='auto', **config)
        if pipe.generation_config.pad_token_id is None:
            pipe.generation_config.pad_token_id = pipe.tokenizer.eos_token_id
        if pipe.generation_config.temperature == 0 or pipe.generation_config.do_sample == False:
            n_sim = 1
        else:
            n_sim = n_generate
    
    elif question_batch == 'all':
        # API_key = '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' # hong
        API_key = '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' # lim
        client = OpenAI(api_key=API_key)

        with open(path_form57_img, "rb") as f:
            img_file = client.files.create(
                file=f,
                purpose="vision"
            )

    ############ extract information using csv-version json

    question_base = (
        "From the given context, answer each question according to its specified answer type: "
    )
    answer_format = (
        "Do not provide explanations.\n"
        "If no information is available for a question, respond with 'Unknown'."
        "The output format must be:\n"
        '(entry_index): answer'
    )

    for idx_content, row in tqdm(df_retrieval.iterrows(), total=df_retrieval.shape[0]):
        if (row != '').sum() / len(row) > 0.8:
            continue
        title = row['title']
        content = row['content']
        
        if question_batch == 'single':
            for entry_idx in tqdm(dict_form57.keys(), leave=False):
                entry = dict_form57[entry_idx]
                entry_name = entry['name']
                for entry_idx_suffix in dict_idx_ap[entry_idx]:
                    question = question_base + '\n'
                    answer_place = dict_answer_places[entry_idx_suffix]
                    description = f"({answer_place['answer_place_name']})" if len(dict_idx_ap[entry_idx]) > 1 else ''
                    answer_place_info = answer_place['answer_place_info']
                    question += f'({entry_idx_suffix}): ' + entry_name + description + f' {str(answer_place_info)}' + '\n'
                    question += answer_format
                    answer_start_idx = dict_idx_ap[entry_idx][0]
                    prompt = f"Context:\n{content}\n\nQuestion:\n{question}\n\nAnswer:\n({answer_start_idx}):"
                
                    try:
                        output = pipe(prompt, max_new_tokens=30)
                        answers = output[0]['generated_text'].split('Answer:\n')[-1]

                        dict_answer = dict(re.findall(r'\((.+)\):\s*([^\n]+)', answers))

                        # print(question)
                        # print()
                        # print(answers)
                        # print()
                        # pprint(dict_answer)
                        # print()

                        answer = dict_answer.get(entry_idx_suffix, '')
                        df_retrieval.loc[idx_content, entry_idx_suffix] = answer
                            
                    except:
                        pass
                    
        elif question_batch == 'group':
            for group_name, group in tqdm(dict_form57_group.items(), leave=False):
                question = question_base + '\n'
                for entry_idx in group:
                    entry = dict_form57[entry_idx]
                    entry_name = entry['name']
                    for entry_idx_suffix in dict_idx_ap[entry_idx]:
                        answer_place = dict_answer_places[entry_idx_suffix]
                        description = f"({answer_place['answer_place_name']})" if len(dict_idx_ap[entry_idx]) > 1 else ''
                        answer_place_info = answer_place['answer_place_info']
                        # question += f'({entry_idx}{entry_idx_suffix}): ' + entry_name + f'({answer_place_name})' + f' {str(answer_place_info)}' + '\n'
                        question += f'({entry_idx_suffix}): ' + entry_name + description + f' {str(answer_place_info)}' + '\n'
                question += answer_format
                answer_start_idx = dict_idx_ap[group[0]][0]
                # prompt = f"Context:\n{title}\n{content}\n\nQuestion:\n{question}\n\nAnswer:\n"
                prompt = f"Context:\n{content}\n\nQuestion:\n{question}\n\nAnswer:\n({answer_start_idx}):"
                
                try:
                    output = pipe(prompt)#, max_new_tokens=30)
                    answers = output[0]['generated_text'].split('Answer:\n')[-1]

                    dict_answer = dict(re.findall(r'\((.+)\):\s*([^\n]+)', answers))

                    # print(question)
                    # print()
                    # print(answer)
                    # print()
                    # pprint(dict_answer)
                    # print()

                    list_entry_idx_suffix = list(chain.from_iterable([dict_idx_ap[entry_idx] for entry_idx in group]))
                    for entry_idx_suffix in list_entry_idx_suffix:
                        answer = dict_answer.get(entry_idx_suffix, '')
                        df_retrieval.loc[idx_content, entry_idx_suffix] = answer
                        
                except:
                    pass
            # df_retrieval.loc[idx_content, :].to_dict()

        elif question_batch == 'all':
            question_base = (
                "From the given context, answer each field in the form.\n"
                "If a field is single-choice type, answer only with the choice code.\n"
                f"The field indices to be answered are: {', '.join(list(map(lambda x: f'({x})', list_answer_places)))}.\n"
            )
            answer_format = (
                "Do not provide explanations.\n"
                "If no information is available for a question, respond with 'Unknown'."
                "The output format must be in JSON:\n"
                '```{"entry_index": "answer"}```\n'
            )
            question = question_base + answer_format
            prompt = f"Context:\n{content}\n\nQuestion:\n{question}"

            content = [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "file_id": img_file.id},
            ]
            messages = [
                {
                    "role": "user",
                    "content": content,
                },
            ]
            response = client.responses.create(model='o4-mini', input=messages)
            output = response.output_text
            dict_answer = parse_json_from_output(output)
            
            for entry_idx, answer in dict_answer.items():
                if entry_idx in list_answer_places:
                    try:
                        df_retrieval.loc[idx_content, entry_idx] = answer
                    except:
                        pass
        
        if idx_content % 10 == 0:
            df_retrieval.to_csv(path_df_form57_retrieval, index=False)

    df_retrieval.to_csv(path_df_form57_retrieval, index=False)

    if question_batch in ['single', 'group']:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    return df_retrieval