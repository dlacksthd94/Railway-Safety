import pandas as pd
import json
from transformers import pipeline, set_seed
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
from google import genai
from .utils import (parse_json_from_output, desanitize_model_path, 
                    generate_openai, generate_hf, select_generate_func, 
                    prepare_df_match, prepare_dict_form57, prepare_dict_form57_group)


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


def extract_helper(api, news_content, question, generate_func, generator, model_path, generation_config=None):
    if api == 'Huggingface':
        # prompt = f"Context:\n{news_content}\n\nQuestion:\n{question}\n\nAnswer:\n({answer_start_idx}):"
        prompt = f"Context:\n{news_content}\n\nQuestion:\n{question}\n\nAnswer:\n"
        content = prompt
        answers = generate_func(generator, model_path, content, generation_config)

    elif api == 'OpenAI':
        prompt = f"Context:\n{news_content}\n\nQuestion:\n{question}"
        content = [
            {"type": "input_text", "text": prompt},
        ]
        answers = generate_func(generator, model_path, content, generation_config=None)
    
    elif api == 'Google':
        prompt = f"Context:\n{news_content}\n\nQuestion:\n{question}"
        content = [prompt]
        answers = generate_func(generator, model_path, content, generation_config=None)    
    else:
        raise ValueError(f'{api} is not supported here.')
    
    # dict_answer = dict(re.findall(r'\((.+)\):\s*([^\n]+)', answers))
    dict_answer = parse_json_from_output(answers)

    return dict_answer


def extract_keywords(cfg):
    _, _, _, json_source, seed = cfg.conv.to_tuple()
    api, model_path, n_generate, question_batch = cfg.retr.to_tuple()
    model_path = desanitize_model_path(model_path)

    set_seed(seed)

    # if json_source == 'None':
    #     # list_answer_places = ["1", "2", "3", "4", "5_month", "5_day", "5_year", "6", "6_ampm", "7", "8", "9", "10", "11", "12", "12_ownership", 
    #     #                       "13", "14", "15", "16", "17", "18", "19", "20a", "20b", "20c", 
    #     #                       "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "30_record", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", 
    #     #                       "44", "45", "46_killed", "46_injured", "47", "48", "49_killed", "49_injured", "50", "51", "52_killed", "52_injured", "53a", "53b", "54", "55", "56", "57"]
    # else:
    dict_form57 = prepare_dict_form57(cfg)
    dict_answer_places, dict_idx_ap = to_answer_places(dict_form57)
    list_answer_places = list(dict_answer_places.keys())

    with open(cfg.path.dict_answer_places, 'w') as f:
        json.dump(dict_answer_places, f, indent=4)

    if question_batch == 'group':
        dict_form57_group = prepare_dict_form57_group(cfg)
    else:
        dict_form57_group = {}

    if os.path.exists(cfg.path.df_retrieval):
        df_retrieval = pd.read_csv(cfg.path.df_retrieval)
        df_retrieval = df_retrieval.fillna('')
    elif os.path.exists(cfg.path.df_match):
        df_match = prepare_df_match(cfg)
        df_retrieval = df_match[['news_id', 'url', 'pub_date', 'title', 'content']].copy(deep=True)
        df_retrieval[list_answer_places] = ''
    else:
        df_news_articles_filter = pd.read_csv(cfg.path.df_news_articles_filter)
        df_retrieval = df_news_articles_filter.copy(deep=True)
        df_retrieval[list_answer_places] = ''
    
    if api == 'Huggingface':
        dict_model_config = {
            'microsoft/phi-4': {},
            'Qwen/Qwen2.5-VL-72B-Instruct': {'load_in_4bit': True},
        }
        quant_config = dict_model_config[model_path]
        
        config = {
            'max_new_tokens': 1024,
            'truncation': True,
            'use_cache': True,
            'do_sample': False,
        }

        pipe = pipeline(model=model_path, device_map='auto', model_kwargs=quant_config, **config)
        if pipe.generation_config:
            if pipe.generation_config.pad_token_id is None and pipe.tokenizer:
                pipe.generation_config.pad_token_id = pipe.tokenizer.eos_token_id
            if pipe.generation_config.temperature == 0 or pipe.generation_config.do_sample == False:
                n_sim = 1
            else:
                n_sim = n_generate
        
        image = Image.open(cfg.path.form57_img)
        img_obj = image
        generator = pipe
    
    elif api == 'OpenAI':
        API_key = cfg.apikey.openai
        client = OpenAI(api_key=API_key)

        with open(cfg.path.form57_img, "rb") as f:
            img_file = client.files.create(
                file=f,
                purpose="vision"
            )
        img_obj = img_file
        generator = client

    elif api == 'Google':
        API_key = cfg.apikey.google
        model_path = desanitize_model_path(model_path)
        client = genai.Client(api_key=API_key)

        generator = client
    
    else:
        raise ValueError(f'{api} is not supported here.')
    
    generate_func = select_generate_func(api)

    ############ extract information using csv-version json

    question_base = (
        "From the given context, answer each question according to its specified answer type: "
    )
    answer_format = (
        "Do not provide explanations.\n"
        "If a field is single-choice type, answer only with the choice code.\n"
        "If no information is available for a question, respond with 'Unknown'.\n"
        "The output format must be in JSON:\n"
        '```{"entry_index": "answer"}```\n'
    )

    for idx_content, row in tqdm(df_retrieval.iterrows(), total=df_retrieval.shape[0]):
        if (row != '').sum() / len(row) > 0.5:
            continue
        title = row['title']
        news_content = row['content']
        
        if question_batch == 'single':
            for entry_idx in tqdm(dict_form57.keys(), leave=False):
                if entry_idx in ['54', '105']:
                    continue
                entry = dict_form57[entry_idx]
                entry_name = entry['name']
                for entry_idx_suffix in dict_idx_ap[entry_idx]:
                    question = question_base + '\n'
                    answer_place = dict_answer_places[entry_idx_suffix]
                    description = f"({answer_place['answer_place_name']})" if len(dict_idx_ap[entry_idx]) > 1 else ''
                    answer_place_info = answer_place['answer_place_info']
                    # question += f'({entry_idx_suffix}): ' + entry_name + description + f' {str(answer_place_info)}' + '\n'
                    question += f'{entry_idx_suffix}: ' + entry_name + description + f' {str(answer_place_info)}' + '\n'
                    question += answer_format
                    answer_start_idx = dict_idx_ap[entry_idx][0]
                
                    try:
                        generation_config = {'max_new_tokens': 30} # will be used only for Huggingface
                        dict_answer = extract_helper(api, news_content, question, generate_func, generator, model_path, generation_config)

                        # print(question)
                        # print()
                        # print(answers)
                        # print()
                        # pprint(dict_answer)
                        # print()

                        answer = dict_answer.get(entry_idx_suffix, '')
                        df_retrieval.loc[idx_content, entry_idx_suffix] = answer # type: ignore
                            
                    except:
                        pass
                    
        elif question_batch == 'group':
            for group_name, group in tqdm(dict_form57_group.items(), leave=False):
                question = question_base + '\n'
                for entry_idx in group:
                    if entry_idx in ['54', '105']:
                        continue
                    entry = dict_form57[entry_idx]
                    entry_name = entry['name']
                    for entry_idx_suffix in dict_idx_ap[entry_idx]:
                        answer_place = dict_answer_places[entry_idx_suffix]
                        description = f"({answer_place['answer_place_name']})" if len(dict_idx_ap[entry_idx]) > 1 else ''
                        answer_place_info = answer_place['answer_place_info']
                        # question += f'({entry_idx_suffix}): ' + entry_name + description + f' {str(answer_place_info)}' + '\n'
                        question += f'{entry_idx_suffix}: ' + entry_name + description + f' {str(answer_place_info)}' + '\n'
                question += answer_format
                answer_start_idx = dict_idx_ap[group[0]][0]
                
                try:
                    generation_config = {}
                    dict_answer = extract_helper(api, news_content, question, generate_func, generator, model_path, generation_config)

                    # print(question)
                    # print()
                    # print(answer)
                    # print()
                    # pprint(dict_answer)
                    # print()

                    list_entry_idx_suffix = list(chain.from_iterable([dict_idx_ap[entry_idx] for entry_idx in group]))
                    for entry_idx_suffix in list_entry_idx_suffix:
                        answer = dict_answer.get(entry_idx_suffix, '')
                        df_retrieval.loc[idx_content, entry_idx_suffix] = answer # type: ignore
                        
                except:
                    pass
            # df_retrieval.loc[idx_content, :].to_dict()

        elif question_batch == 'all':
            question = question_base + '\n'
            for entry_idx in tqdm(dict_form57.keys(), leave=False):
                if entry_idx in ['54', '105']:
                    continue
                entry = dict_form57[entry_idx]
                entry_name = entry['name']
                for entry_idx_suffix in dict_idx_ap[entry_idx]:
                    answer_place = dict_answer_places[entry_idx_suffix]
                    description = f"({answer_place['answer_place_name']})" if len(dict_idx_ap[entry_idx]) > 1 else ''
                    answer_place_info = answer_place['answer_place_info']
                    # question += f'({entry_idx_suffix}): ' + entry_name + description + f' {str(answer_place_info)}' + '\n'
                    question += f'{entry_idx_suffix}: ' + entry_name + description + f' {str(answer_place_info)}' + '\n'
            question += answer_format
            answer_start_idx = dict_idx_ap[list(dict_form57.keys())[0]][0]
            
            try:
                generation_config = {}
                dict_answer = extract_helper(api, news_content, question, generate_func, generator, model_path, generation_config)

                list_entry_idx_suffix = list(chain.from_iterable([dict_idx_ap[entry_idx] for entry_idx in dict_form57]))
                for entry_idx_suffix in list_entry_idx_suffix:
                    answer = dict_answer.get(entry_idx_suffix, '')
                    df_retrieval.loc[idx_content, entry_idx_suffix] = answer # type: ignore
            
            except:
                pass
            
        if idx_content % 10 == 0: # type: ignore
            df_retrieval.to_csv(cfg.path.df_retrieval, index=False)

    df_retrieval.to_csv(cfg.path.df_retrieval, index=False)

    if api == 'Huggingface':
        del generator
        gc.collect()
        torch.cuda.empty_cache()

    return df_retrieval


def extract_keywords_realtime(cfg):
    _, _, _, json_source, seed = cfg.conv.to_tuple()
    api, model_path, n_generate, question_batch = cfg.retr.to_tuple()
    model_path = desanitize_model_path(model_path)

    set_seed(seed)

    dict_form57 = prepare_dict_form57(cfg)
    dict_answer_places, dict_idx_ap = to_answer_places(dict_form57)
    list_answer_places = list(dict_answer_places.keys())

    with open(cfg.path.dict_answer_places, 'w') as f:
        json.dump(dict_answer_places, f, indent=4)

    if question_batch == 'group':
        dict_form57_group = prepare_dict_form57_group(cfg)
    else:
        dict_form57_group = {}

    if os.path.exists(cfg.path.df_retrieval_realtime):
        df_retrieval = pd.read_csv(cfg.path.df_retrieval_realtime, parse_dates=['pub_date', 'accident_date'])
        df_retrieval = df_retrieval.fillna('')
        
        df_news_articles_filter = pd.read_csv(cfg.path.df_news_articles_realtime_filter, parse_dates=['pub_date', 'accident_date'])
        df_news_articles_new = df_news_articles_filter[~df_news_articles_filter['news_id'].isin(df_retrieval['news_id'])]
        df_news_articles_new.loc[:, list_answer_places] = ''
        df_retrieval = pd.concat([df_retrieval, df_news_articles_new], ignore_index=True)
    else:
        df_news_articles_filter = pd.read_csv(cfg.path.df_news_articles_realtime_filter, parse_dates=['pub_date', 'accident_date'])
        df_retrieval = df_news_articles_filter.copy(deep=True)
        df_retrieval.loc[:, list_answer_places] = ''
        df_retrieval.to_csv(cfg.path.df_retrieval_realtime, index=False)
    
    if api == 'Huggingface':
        dict_model_config = {
            'microsoft/phi-4': {},
            'Qwen/Qwen2.5-VL-72B-Instruct': {'load_in_4bit': True},
        }
        quant_config = dict_model_config[model_path]
        
        config = {
            'max_new_tokens': 1024,
            'truncation': True,
            'use_cache': True,
            'do_sample': False,
        }

        pipe = pipeline(model=model_path, device_map='auto', model_kwargs=quant_config, **config)
        if pipe.generation_config:
            if pipe.generation_config.pad_token_id is None and pipe.tokenizer:
                pipe.generation_config.pad_token_id = pipe.tokenizer.eos_token_id
            if pipe.generation_config.temperature == 0 or pipe.generation_config.do_sample == False:
                n_sim = 1
            else:
                n_sim = n_generate
        
        image = Image.open(cfg.path.form57_img)
        img_obj = image
        generator = pipe
    
    elif api == 'OpenAI':
        API_key = cfg.apikey.openai
        client = OpenAI(api_key=API_key)

        with open(cfg.path.form57_img, "rb") as f:
            img_file = client.files.create(
                file=f,
                purpose="vision"
            )
        img_obj = img_file
        generator = client

    elif api == 'Google':
        API_key = cfg.apikey.google
        model_path = desanitize_model_path(model_path)
        client = genai.Client(api_key=API_key)

        generator = client
    
    else:
        raise ValueError(f'{api} is not supported here.')
    
    generate_func = select_generate_func(api)

    ############ extract information using csv-version json

    question_base = (
        "From the given context, answer each question according to its specified answer type: "
    )
    answer_format = (
        "Do not provide explanations.\n"
        "If a field is single-choice type, answer only with the choice code.\n"
        "If no information is available for a question, respond with 'Unknown'.\n"
        "The output format must be in JSON:\n"
        '```{"entry_index": "answer"}```\n'
    )

    idx_content = df_retrieval.columns.get_loc('content')
    for idx_content, row in tqdm(df_retrieval.iterrows(), total=df_retrieval.shape[0]):
        if (row.iloc[idx_content + 1:] != '').sum() / len(row) > 0.2: # type: ignore
            continue
        title = row['title']
        news_content = row['content']
        
        if question_batch == 'single':
            raise NotImplementedError("Single question batch is not supported in realtime extraction.")
                    
        elif question_batch == 'group':
            for group_name, group in tqdm(dict_form57_group.items(), leave=False):
                question = question_base + '\n'
                for entry_idx in group:
                    if entry_idx in ['105']:
                        continue
                    entry = dict_form57[entry_idx]
                    entry_name = entry['name']
                    for entry_idx_suffix in dict_idx_ap[entry_idx]:
                        answer_place = dict_answer_places[entry_idx_suffix]
                        if entry_idx_suffix == '54':
                            description = f"(summary of the train accident in two sentences)"
                        else:
                            description = f"({answer_place['answer_place_name']})" if len(dict_idx_ap[entry_idx]) > 1 else ''
                        answer_place_info = answer_place['answer_place_info']
                        # question += f'({entry_idx_suffix}): ' + entry_name + description + f' {str(answer_place_info)}' + '\n'
                        question += f'{entry_idx_suffix}: ' + entry_name + description + f' {str(answer_place_info)}' + '\n'
                question += answer_format
                answer_start_idx = dict_idx_ap[group[0]][0]
                
                try:
                    generation_config = {}
                    dict_answer = extract_helper(api, news_content, question, generate_func, generator, model_path, generation_config)

                    # print(question)
                    # print()
                    # print(answer)
                    # print()
                    # pprint(dict_answer)
                    # print()

                    list_entry_idx_suffix = list(chain.from_iterable([dict_idx_ap[entry_idx] for entry_idx in group]))
                    for entry_idx_suffix in list_entry_idx_suffix:
                        answer = dict_answer.get(entry_idx_suffix, '')
                        df_retrieval.loc[idx_content, entry_idx_suffix] = answer # type: ignore
                        
                except:
                    pass
            # df_retrieval.loc[idx_content, :].to_dict()

        elif question_batch == 'all':
            question = question_base + '\n'
            for entry_idx in tqdm(dict_form57.keys(), leave=False):
                if entry_idx in ['105']:
                    continue
                entry = dict_form57[entry_idx]
                entry_name = entry['name']
                for entry_idx_suffix in dict_idx_ap[entry_idx]:
                    answer_place = dict_answer_places[entry_idx_suffix]
                    if entry_idx_suffix == '54':
                        description = f"(summary of the train accident in two sentences)"
                    else:
                        description = f"({answer_place['answer_place_name']})" if len(dict_idx_ap[entry_idx]) > 1 else ''
                    answer_place_info = answer_place['answer_place_info']
                    # question += f'({entry_idx_suffix}): ' + entry_name + description + f' {str(answer_place_info)}' + '\n'
                    question += f'{entry_idx_suffix}: ' + entry_name + description + f' {str(answer_place_info)}' + '\n'
            question += answer_format
            answer_start_idx = dict_idx_ap[list(dict_form57.keys())[0]][0]
            
            try:
                generation_config = {}
                dict_answer = extract_helper(api, news_content, question, generate_func, generator, model_path, generation_config)

                list_entry_idx_suffix = list(chain.from_iterable([dict_idx_ap[entry_idx] for entry_idx in dict_form57]))
                for entry_idx_suffix in list_entry_idx_suffix:
                    answer = dict_answer.get(entry_idx_suffix, '')
                    df_retrieval.loc[idx_content, entry_idx_suffix] = answer # type: ignore
            
            except:
                pass
            
        if idx_content % 10 == 0: # type: ignore
            df_retrieval.to_csv(cfg.path.df_retrieval_realtime, index=False)

    df_retrieval.to_csv(cfg.path.df_retrieval_realtime, index=False)

    if api == 'Huggingface':
        del generator
        gc.collect()
        torch.cuda.empty_cache()

    return df_retrieval


# if __name__ == '__main__':
#     df_retrieval = pd.read_csv(cfg.path.df_retrieval)
#     list_col_new = ['news_id', 'url', 'pub_date', 'title', 'content', 
#                     '1', '2', '3', '4', '5_1', '5_2', '5_3', '6_1', '6_2', '7', '8', '9', '10', '11', '12_1', '12_2', 
#                     '13', '14', '15', '16', '17', '18', '19', '20a', '20b', '20c', 
#                     '21', '22', '23', '24', '25', '26', '27', '28', '29', '30_1', '30_2', '31', 
#                     '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', 
#                     '46_1', '46_2', '47', '48', '49_1', '49_2', '50', '51', '52_1', '52_2', '53a', '53b', '54', '55', '56', '57']
#     df_retrieval.columns = list_col_new
#     df_retrieval.to_csv(cfg.path.df_retrieval, index=False)

#     if api == 'Huggingface':
#         prompt = f"Context:\n{news_content}\n\nQuestion:\n{question}\n\nAnswer:\n"
#         if model_path == 'microsoft/phi-4':
#             content = prompt
#             generation_config = {}
#             answers = generate_func(generator, model_path, content, generation_config)
#         elif model_path == 'Qwen/Qwen2.5-VL-72B-Instruct':
#             content = [
#                 {"type": "image", "image": img_obj},
#                 {"type": "text", "text": prompt},
#             ]
#             generation_config = {}
#             answers = generate_func(generator, model_path, content, generation_config)
#         else:
#             raise ValueError(f'{model_path} is not supported here.')

#     elif api == 'OpenAI':
#         prompt = f"Context:\n{news_content}\n\nQuestion:\n{question}"
#         content = [
#             {"type": "input_text", "text": prompt},
#             {"type": "input_image", "file_id": img_obj.id}, # type: ignore
#         ]
#         answers = generate_func(generator, model_path, content, generation_config=None)
    
#     else:
#         raise ValueError(f'{api} is not supported here.')