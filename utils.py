import os
import time
import utils_scrape
import json5
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import Counter
from itertools import chain
from transformers import pipeline
import datetime
import gc
import torch
import json
import ast
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self._start = time.perf_counter()
        return self           # (optional) so you can read .elapsed later
    
    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self._start
        elapsed = self.format_hms(self.elapsed)
        print(f"[{self.label}]\t elapsed: {elapsed}")
            
    def format_hms(self, total_seconds):
        h = int(total_seconds // 3600)
        m = int((total_seconds % 3600) // 60)
        s = total_seconds % 60               # still a float now
        return f"{h:02d}:{m:02d}:{s:06.3f}"   # e.g. 00:01:02.357

def as_float(val):
    try:
        return float(val)
    except:
        return val

def lower_str(val):
    if isinstance(val, str):
        return val.lower()
    else:
        return val

def parse_json_from_output(output):
    """Parse JSON from the output text of OpenAI response.
    If the output is a plain JSON string, it parses that directly.
    Otherwise, find the code block containing JSON and parse it.
    """
    try:
        if '```' in output:
            json_start_index = output.index('```')
            json_end_index = output.rindex('```')
            str_form57 = output[json_start_index:json_end_index].strip('`')
            if str_form57.startswith('json'):
                str_form57 = str_form57.replace('json', '', 1)
        else:
            str_form57 = output
        try:
            dict_form57 = json.loads(str_form57)
        except:
            dict_form57 = ast.literal_eval(str_form57)
    except:
        dict_form57 = {}
    return dict_form57

def text_binary_classification(pipe, prompt, dict_answer_choice, num_sim):
    list_output = pipe(prompt, max_new_tokens=1, num_return_sequences=num_sim, return_full_text=False)
    list_answer = list(map(lambda output: output['generated_text'].upper(), list_output))
    list_answer_filter = list(filter(lambda answer: answer in dict_answer_choice, list_answer))
    list_answer_map = list(map(lambda answer: dict_answer_choice[answer], list_answer_filter))
    return list_answer_map

def text_generation(pipe, prompt, max_new_tokens=4096):
    output = pipe(prompt, max_new_tokens=max_new_tokens, return_full_text=False)
    answer = output[0]['generated_text']
    return answer

def get_acc_table(path_form57_csv, path_dict_col_indexing, path_dict_idx_mapping, path_dict_answer_places, df_merge, list_answer_type_selected, config):
    _, _, _, json_source = config.conversion.to_tuple()
    _, model, _, _ = config.retrieval.to_tuple()

    df_form57_csv = pd.read_csv(path_form57_csv)
    df_form57_csv = df_form57_csv[df_form57_csv['State Name'] == 'CALIFORNIA']
    df_form57_csv['Date'] = pd.to_datetime(df_form57_csv['Date'])
    df_form57_csv = df_form57_csv[df_form57_csv['Date'] >= '2000-01-01']
    df_form57_csv['Date'] = df_form57_csv['Date'].astype(str)
    df_form57_csv = df_form57_csv.set_index('Report Key')

    with open(path_dict_col_indexing, 'r') as f:
        dict_col_indexing = json5.load(f)
    
    with open(path_dict_idx_mapping, 'r') as f:
        dict_idx_mapping = json5.load(f)
        dict_idx_mapping_inverse = {v: k for k, v in dict_idx_mapping.items()}
        if '' in dict_idx_mapping_inverse:
            dict_idx_mapping_inverse.pop('')

    with open(path_dict_answer_places, 'r') as f:
        dict_answer_places = json5.load(f)
    
    dict_idx_answer_type = {
        'digit': ['5_month', '5_day', '5_year', '6_hour', '6_minute', '14', '18', '20c_quantity', '21', '28', '29', '30', '38', 
                    '46_killed', '46_injured', '47', '48', '49_killed', '49_injured', '50', '52_killed', '52_injured'],
        'text': ['1', '2', '3', '5', '6', '7', '9', '11', '12', '20c_name', '20c_measure', '26'],
        'choice': ['6_ampm', '12_ownership', '13', '15', '16', '17', '19', '20a', '20b', '22', '23', '24', '25', '30_record', '31', 
                    '34', '35', '36', '37', '39', '40', '41', '42', '43', '44', '45'],
        'etc': ['8', '10', '27'],
    }
    dict_idx_answer_type = {k: {idx: dict_col_indexing[idx] for idx in v} for k, v in dict_idx_answer_type.items()}
    # elif json_source == 'csv':
    # dict_idx_answer_type = {
    #     'digit': {'15': 'Month', '16': 'Day', '3': 'Incident Year', '17': 'Hour', '18': 'Minute',
    #               '30': 'Estimated Vehicle Speed', '34': 'Railroad Car Unit Position', '39': 'Hazmat Released Quantity',
    #               '41': 'Temperature', '48': 'Number of Locomotive Units', '49': 'Number of Cars', '50': 'Train Speed',
    #               '83': 'User Age', '91': 'Crossing Users Killed', '92': 'Crossing Users Injured',
    #               '93': 'Vehicle Damage Cost', '94': 'Number Vehicle Occupants', '95': 'Employees Killed', '96': 'Employees Injured',
    #               '97': 'Number People On Train', '99': 'Passengers Killed', '100': 'Passengers Injured'},
    #     'text': {'1': 'Railroad Name', '5': 'Other Railroad Name', '9': 'Maintenance Railroad Name', '14': 'Date', '20': 'Time',
    #              '21': 'Nearest Station', '24': 'County Name', '26': 'City Name', '27': 'Highway Name',
    #              '38': 'Hazmat Released Name', '40': 'Hazmat Released Measure', '46': 'Track Name'},
    #     'choice': {'19': 'AM/PM', '28': 'Public/Private', '29': 'Highway User Code', '31': 'Vehicle Direction Code', '32': 'Highway User Position Code',
    #                '33': 'Equipment Involved Code', '35': 'Equipment Struck Code',
    #                '36': 'Hazmat Involvement Code', '37': 'Hazmat Released by Code',
    #                '42': 'Visibility Code', '43': 'Weather Condition Code',
    #                '44': 'Equipment Type Code', '45': 'Track Type Code', '51': 'Estimated/Recorded Speed', '52': 'Train Direction Code',
    #                '79': 'Roadway Condition Code', '80': 'Crossing Warning Location Code', '81': 'Warning Connected To Signal', '82': 'Crossing Illuminated',
    #                '84': 'User Sex', '85': 'User Struck By Second Train', '86': 'Highway User Action Code', '87': 'Driver Passed Vehicle',
    #                '88': 'View Obstruction Code', '89': 'Driver Condition Code', '90': 'Driver In Vehicle'},
    #     'etc': {'8': 'Subdivision', '25': 'State Name', '47': 'Track Class'}
    # }

    # sanity check
    list_all = list(chain.from_iterable(map(set, dict_idx_answer_type.values())))
    counts = Counter(list_all)
    duplicates = [item for item, count in counts.items() if count > 1]
    assert len(duplicates) == 0

    dict_idx_selected = {dict_idx_mapping[k]: k for answer_type_selected in list_answer_type_selected for k, v in dict_idx_answer_type[answer_type_selected].items()}
    if '' in dict_idx_selected:
        dict_idx_selected.pop('')

    pipe = pipeline(model='microsoft/phi-4', device_map='auto', **{'do_sample': False})

    df_acc = df_merge.copy(deep=True)
    df_acc = df_acc.drop('match', axis=1, errors='ignore')
    df_acc.iloc[:, 12:] = np.nan
    for idx_match, row_match in tqdm(df_merge.iterrows(), total=df_merge.shape[0]):
        report_key = row_match['report_key']
        row_csv = df_form57_csv.loc[report_key]
        list_score_temp = []
        for col_idx_json, col_idx_form  in dict_idx_selected.items():
            col_name = dict_col_indexing[col_idx_form]
            # if col_idx_json not in row_match:
            #     continue
            retrieval = row_match[col_idx_json]
            label = row_csv[col_name]
            if isinstance(retrieval, str):
                retrieval = retrieval.strip('\'"').strip()
            acc_temp = np.nan
            if not (isinstance(retrieval, str) and retrieval.lower() == 'unknown') and pd.notna(retrieval) and pd.notna(label):
                if col_idx_form in dict_idx_answer_type['text']:
                    dict_answer_choice = {'YES': True, 'NO': False}
                    context = f"ground-truth label: {label.lower()}\ninference output: {retrieval.lower()}"
                    if col_name == 'Time':
                        question = f"Is the inferred time within a tolerance of ±1 hour of the ground-truth time, disregarding AM/PM? Answer only { '/'.join(dict_answer_choice) }."
                    elif col_name == 'Highway Name':
                        question = f"Do any of the inferred highway names fuzzy-match the ground-truth highway names? Answer only { '/'.join(dict_answer_choice) }." # 10
                    else:
                        question = f"Does the inference output fuzzy-match the ground-truth label? Answer only {'/'.join(dict_answer_choice)}" # 23
                    prompt = f"{context}\n\nQuestion:\n{question}\n\nAnswer:\n"
                    acc_temp = text_binary_classification(pipe, prompt, dict_answer_choice, num_sim=1)[0]
                elif col_idx_form in dict_idx_answer_type['digit']:
                    retrieval = as_float(retrieval)
                    acc_temp = (retrieval == label)
                    if np.issubdtype(np.array(retrieval).dtype, np.number):
                        if col_name == 'Incident Year':
                            retrieval_dt = datetime.datetime(year=int(retrieval), month=1, day=1).strftime('%y')
                            label_dt = datetime.datetime(year=int(label), month=1, day=1).strftime('%y')
                            acc_temp = retrieval_dt == label_dt
                        # elif col_name == 'Day':
                        #     acc_temp = abs(retrieval - label) <= 1
                        elif col_name in ['User Age', 'Train Speed', 'Estimated Vehicle Speed', 'Number People On Train']:
                            acc_temp = abs(retrieval - label) <= 10
                elif col_idx_form in dict_idx_answer_type['choice']:
                    choices = dict_answer_places[col_idx_json]['answer_place_info']['choices']
                    retrieval_choie_key = retrieval.upper() if isinstance(retrieval, str) else str(int(retrieval)).upper()
                    retrieval_choice_value = choices.get(retrieval_choie_key, '').upper()
                    label_choice_key = label.upper() if isinstance(label, str) else str(int(label))
                    label_choice_value = choices.get(label_choice_key, '').upper()
                    
                    acc_temp_by_key_key = (retrieval_choie_key == label_choice_key)
                    acc_temp_by_key_value = (retrieval_choie_key == label_choice_value)
                    acc_temp_by_value_key = (retrieval_choice_value == label_choice_key)
                    acc_temp = any([acc_temp_by_key_key, acc_temp_by_key_value, acc_temp_by_value_key])
                elif col_idx_form in dict_idx_answer_type['etc']:
                    pass
                else:
                    raise f"no col like {col_idx_form}: {col_name}"
            list_score_temp.append(acc_temp)
        #     print(f'{acc_temp}\t{retrieval}=={label}\t{col_name}')
        # print('--------------------------------------------------')
        df_acc.loc[idx_match, list(dict_idx_selected.keys())] = list_score_temp
    
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return df_acc

def get_cov_table(path_dict_col_indexing, path_dict_idx_mapping, path_dict_answer_places, df_retrieval, df_annotate, list_answer_type_selected, config):
    _, _, _, json_source = config.conversion.to_tuple()
    _, model, _, _ = config.retrieval.to_tuple()

    with open(path_dict_col_indexing, 'r') as f:
        dict_col_indexing = json5.load(f)
    
    with open(path_dict_idx_mapping, 'r') as f:
        dict_idx_mapping = json5.load(f)
        dict_idx_mapping_inverse = {v: k for k, v in dict_idx_mapping.items()}
        if '' in dict_idx_mapping_inverse:
            dict_idx_mapping_inverse.pop('')

    with open(path_dict_answer_places, 'r') as f:
        dict_answer_places = json5.load(f)
    
    dict_idx_answer_type = {
        'digit': ['5_month', '5_day', '5_year', '6_hour', '6_minute', '14', '18', '20c_quantity', '21', '28', '29', '30', '38', 
                    '46_killed', '46_injured', '47', '48', '49_killed', '49_injured', '50', '52_killed', '52_injured'],
        'text': ['1', '2', '3', '5', '6', '7', '9', '11', '12', '20c_name', '20c_measure', '26'],
        'choice': ['6_ampm', '12_ownership', '13', '15', '16', '17', '19', '20a', '20b', '22', '23', '24', '25', '30_record', '31', 
                    '34', '35', '36', '37', '39', '40', '41', '42', '43', '44', '45'],
        'etc': ['8', '10', '27'],
    }
    dict_idx_answer_type = {k: {idx: dict_col_indexing[idx] for idx in v} for k, v in dict_idx_answer_type.items()}

    # sanity check
    list_all = list(chain.from_iterable(map(set, dict_idx_answer_type.values())))
    counts = Counter(list_all)
    duplicates = [item for item, count in counts.items() if count > 1]
    assert len(duplicates) == 0

    dict_idx_selected = {dict_idx_mapping[k]: k for answer_type_selected in list_answer_type_selected for k, v in dict_idx_answer_type[answer_type_selected].items()}
    if '' in dict_idx_selected:
        dict_idx_selected.pop('')
    if '20c' in dict_idx_selected:
        dict_idx_selected.pop('20c')

    df_annotate = df_annotate.drop('annotated', axis=1, errors='ignore')
    df_cov = df_annotate.copy(deep=True)
    idx_news_content = df_cov.columns.get_loc('content')
    df_cov = df_cov.iloc[:, :idx_news_content + 1]
    df_cov['fnr'] = np.nan

    list_col_idx_form = df_annotate.columns[idx_news_content + 1:]
    
    for idx_annotate, row_annotate in tqdm(df_annotate.iterrows(), total=df_annotate.shape[0]):
        news_id = row_annotate['news_id']
        row_retrieval = df_retrieval[df_retrieval['news_id'] == news_id].squeeze()
        list_attempt = []
        list_label = []
        for col_idx_form in list_col_idx_form:
            try:
                col_idx_json = dict_idx_mapping[col_idx_form]
                pred = str(row_retrieval[col_idx_json]).lower() != 'unknown'
                label = row_annotate[col_idx_form] == 'True'
                list_attempt.append(pred)
                list_label.append(label)
            except:
                pass
        
        # cov_temp = f1_score(list_label, list_attempt, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(list_label, list_attempt).ravel()
        cov_temp = tp / (fn + tp) if (fn + tp) > 0 else np.nan
        df_cov.loc[idx_annotate, 'fnr'] = cov_temp
    
    return df_cov