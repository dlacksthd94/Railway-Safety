import pandas as pd
import json5
import numpy as np
from tqdm import tqdm
from itertools import chain
from collections import Counter
import datetime
from transformers import pipeline
import gc
import torch
from .utils import (text_binary_classification, as_float, 
                    prepare_dict_col_indexing, prepare_dict_idx_mapping, prepare_dict_answer_places)
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

def get_acc_table(list_answer_type_selected, cfg):
    df_record = pd.read_csv(cfg.path.df_record)
    df_record = df_record[df_record['State Name'].str.title().isin(cfg.scrp.target_states)]
    df_record['Date'] = pd.to_datetime(df_record['Date'])
    df_record = df_record[df_record['Date'] >= cfg.scrp.start_date]

    df_record['Date'] = df_record['Date'].astype(str)
    df_record = df_record.set_index('Report Key')

    df_record_retrieval = pd.read_csv(cfg.path.df_record_retrieval)

    dict_col_indexing = prepare_dict_col_indexing(cfg)
    dict_idx_mapping, dict_idx_mapping_inverse = prepare_dict_idx_mapping(cfg)
    assert len(dict_idx_mapping) == 63
    dict_idx_mapping_cleaned = {k: v for k, v in dict_idx_mapping.items() if v != ''}
    dict_answer_places = prepare_dict_answer_places(cfg)
    
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

    pipe = pipeline(model='microsoft/phi-4', device_map='auto', **{'do_sample': False}) # type: ignore

    idx_content = df_record_retrieval.columns.get_loc('content')
    df_acc = df_record_retrieval.copy(deep=True)
    df_acc = df_acc.iloc[:, :idx_content + 1] # type: ignore
    df_acc[list(dict_idx_mapping_cleaned.keys())] = np.nan
    df_acc = df_acc.drop('match', axis=1, errors='ignore')
    for idx_match, row_match in tqdm(df_record_retrieval.iterrows(), total=df_record_retrieval.shape[0]):
        report_key = row_match['report_key']
        row_csv = df_record.loc[report_key]
        
        list_score_temp = []
        for col_idx_form, col_idx_json in dict_idx_mapping_cleaned.items():
            col_name = dict_col_indexing[col_idx_form]
            if col_idx_json not in row_match.index:
                acc_temp = np.nan
            else:
                retrieval = row_match.loc[col_idx_json]
                label = row_csv.loc[col_name]
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
                                if retrieval:
                                    retrieval_dt = datetime.datetime(year=int(retrieval), month=1, day=1).strftime('%y')
                                else:
                                    retrieval_dt = '0'
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
                        raise BaseException(f"no col like {col_idx_form}: {col_name}")
            list_score_temp.append(acc_temp)
        #     print(f'{acc_temp}\t{retrieval}=={label}\t{col_name}')
        # print('--------------------------------------------------')
        df_acc.loc[idx_match, list(dict_idx_mapping_cleaned.keys())] = list_score_temp # type: ignore
    
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    idx_news_content = df_acc.columns.get_loc('content')
    acc = df_acc.iloc[:, idx_news_content + 1:].dropna(axis=1, how='all').mean().mean() # type: ignore

    return df_acc, acc

def get_cov_table(list_answer_type_selected, cfg):
    df_retrieval = pd.read_csv(cfg.path.df_retrieval)
    df_annotate = pd.read_csv(cfg.path.df_annotate)
    df_annotate = df_annotate[df_annotate['annotated'] == 1]

    dict_idx_mapping, dict_idx_mapping_inverse = prepare_dict_idx_mapping(cfg)
    assert len(dict_idx_mapping) == 63
    dict_idx_mapping_cleaned = {k: v for k, v in dict_idx_mapping.items() if v != ''}

    df_annotate = df_annotate.drop('annotated', axis=1, errors='ignore')
    df_cov = df_annotate.copy(deep=True)
    idx_news_content = df_cov.columns.get_loc('content')
    df_cov = df_cov.iloc[:, :idx_news_content + 1] # type: ignore
    df_cov['fnr'] = np.nan
    
    for idx_annotate, row_annotate in tqdm(df_annotate.iterrows(), total=df_annotate.shape[0]):
        news_id = row_annotate['news_id']
        row_retrieval = df_retrieval[df_retrieval['news_id'] == news_id].squeeze()
        list_attempt = []
        list_label = []
        for col_idx_form, col_idx_json in dict_idx_mapping_cleaned.items():
            try:
                pred = str(row_retrieval[col_idx_json]).lower() != 'unknown'
                label = row_annotate[col_idx_form] == 'True'
                list_attempt.append(pred)
                list_label.append(label)
            except:
                # print(f'{col_idx_form} not in df_annotate')
                pass
        
        # cov_temp = f1_score(list_label, list_attempt, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(list_label, list_attempt).ravel()
        cov_temp = tp / (fn + tp) if (fn + tp) > 0 else np.nan
        df_cov.loc[idx_annotate, 'fnr'] = cov_temp # type: ignore

    idx_news_content = df_cov.columns.get_loc('content')
    cov = df_cov.iloc[:, idx_news_content + 1:].dropna(axis=1, how='all').mean().mean() # type: ignore
    
    return df_cov, cov

def get_stats(df_acc, cfg):
    idx_content = df_acc.columns.get_loc('content')
    most_failed_cnt = (df_acc.iloc[:, idx_content + 1:] == False).sum().sort_values(ascending=False)
    most_failed_acc = 1 - most_failed_cnt / df_acc[most_failed_cnt.index].notna().sum()
    df_most_failed = pd.DataFrame({
        'failed_count': most_failed_cnt,
        'accuracy': most_failed_acc
    })
    print("Top 5 Most Failed Items:", df_most_failed[:5], sep="\n")

    answered_cnt = (df_acc.iloc[:, idx_content + 1:].notna()).sum().sort_values(ascending=False)
    answered_cnt = answered_cnt[answered_cnt >= 50]
    least_acc = most_failed_acc[answered_cnt.index]
    df_least_acc = pd.DataFrame({
        'answered_cnt': answered_cnt,
        'accuracy': least_acc
    })
    df_least_acc = df_least_acc.sort_values('accuracy')
    print("Top 5 Least Accurate Items:", df_least_acc[:5], sep="\n")