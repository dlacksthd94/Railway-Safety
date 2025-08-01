import os
import time
import utils_scrape
import json5
from tqdm import tqdm
import numpy as np
import pandas as pd

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

def get_acc_table(path_form57_csv, path_dict_col_idx_name, df_match, dict_form57, list_answer_type_selected):
    df_form57_csv = pd.read_csv(path_form57_csv)
    df_form57_csv = df_form57_csv[df_form57_csv['State Name'] == 'CALIFORNIA']
    df_form57_csv['hash_id'] = df_form57_csv.apply(utils_scrape.hash_row, axis=1)
    df_form57_csv['Date'] = pd.to_datetime(df_form57_csv['Date'])
    df_form57_csv = df_form57_csv[df_form57_csv['Date'] >= '2000-01-01']

    with open(path_dict_col_idx_name, 'r') as f:
        dict_col_idx_name = json5.load(f)

    dict_idx_answer_type = {
        'digit': {"14": "Estimated Vehicle Speed", "18": "Railroad Car Unit Position", "21": "Temperature",
                  "28": "Number of Locomotive Units", "29": "Number of Cars", "38": "User Age", "47": "Vehicle Damage Cost", "48": "Number Vehicle Occupants", "50": "Number People On Train"},
        'str': {"1": "Railroad Name", "2": "Other Railroad Name", "3": "Maintenance Railroad Name", "7": "Nearest Station", "9": "County Name", "11": "City Name", "26": "Track Name", "27": "Track Class"},
        'list': {"5": ["Date", "Month", "Day", "Incident Year"], "6": ["Time", "Hour", "Minute", "AM/PM"], "12": ["Highway Name", "Public/Private"],
                 "20c": ["Hazmat Released Name", "Hazmat Released Quantity", "Hazmat Released Measure"], "30": ["Train Speed", "Estimated/Recorded Speed"],
                 "46": ["Crossing Users Killed", "Crossing Users Injured"], "49": ["Employees Killed", "Employees Injured"], "52": ["Passengers Killed", "Passengers Injured"]},
        'choice': {"13": "Highway User Code", "15": "Vehicle Direction Code", "16": "Highway User Position Code",
                   "17": "Equipment Involved Code", "19": "Equipment Struck Code",
                   "20a": "Hazmat Involvement Code", "20b": "Hazmat Released by Code",
                   "22": "Visibility Code", "23": "Weather Condition Code",
                   "24": "Equipment Type Code", "25": "Track Type Code", "31": "Train Direction Code",
                   "34": "Roadway Condition Code", "35": "Crossing Warning Location Code", "36": "Warning Connected To Signal", "37": "Crossing Illuminated",
                   "39": "User Sex", "40": "User Struck By Second Train", "41": "Highway User Action Code", "42": "Driver Passed Vehicle", "43": "View Obstruction Code",
                   "44": "Driver Condition Code", "45": "Driver In Vehicle"},
        'etc': {"10": "State Name"}
    }

    # choose only choice-answer idx
    dict_idx_selected = {k: v for answer_type_selected in list_answer_type_selected for k, v in dict_idx_answer_type[answer_type_selected].items()}

    df_acc = df_match.copy(deep=True)
    df_acc = df_acc.drop('match', axis=1, errors='ignore')
    df_acc.loc[:, '1':] = np.nan
    for idx_match, row_match in tqdm(df_match.iterrows(), total=df_match.shape[0]):
        incident_id = row_match['incident_id']
        row_csv = df_form57_csv[df_form57_csv['hash_id'] == incident_id]
        list_score_temp = []
        for col_idx, col_name in dict_idx_selected.items():
            retrieval = row_match[col_idx]
            label = row_csv[col_name]
            if isinstance(retrieval, str):
                retrieval = retrieval.strip('\'"').strip()
            acc_temp = np.nan
            if not (isinstance(retrieval, str) and retrieval.lower() == 'unknown') and pd.notna(retrieval) and label.notna().squeeze():
                if col_idx in dict_idx_answer_type['str']:
                    # must use LLM to check non-choice answer
                    pass
                elif col_idx in dict_idx_answer_type['list']:
                    # must use a fine-grained metric
                    pass
                elif col_idx in dict_idx_answer_type['digit']:
                    pass
                    acc_temp = (retrieval == label.squeeze())
                elif col_idx in dict_idx_answer_type['choice']:
                    choices = dict_form57[col_idx]['choices']
                    retrieval_choie_key = retrieval.upper() if isinstance(retrieval, str) else str(int(retrieval)).upper()
                    retrieval_choice_value = choices.get(retrieval_choie_key, '').upper()
                    label_choice_key = label.squeeze().upper() if isinstance(label.squeeze(), str) else str(int(label.squeeze()))
                    label_choice_value = choices.get(label_choice_key, '').upper()
                    
                    acc_temp_by_key_key = (retrieval_choie_key == label_choice_key)
                    acc_temp_by_key_value = (retrieval_choie_key == label_choice_value)
                    acc_temp_by_value_key = (retrieval_choice_value == label_choice_key)
                    acc_temp = any([acc_temp_by_key_key, acc_temp_by_key_value, acc_temp_by_value_key])
                elif col_idx in dict_idx_answer_type['etc']:
                    pass
                else:
                    raise f"no col like {col_idx}: {col_name}"
            list_score_temp.append(acc_temp)
            print(f'{acc_temp}\t{retrieval}=={label.squeeze()}\t{col_name}')
        # print(row_match['rd_url'])
        print('--------------------------------------------------')
        df_acc.loc[idx_match, list(dict_idx_selected.keys())] = list_score_temp
    return df_acc