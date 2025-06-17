import pandas as pd
import json
import utils_scrape
from transformers import pipeline
import torch
import time
from pprint import pprint
from tqdm import tqdm

DATA_FOLDER = 'data/'
FN_DF_DATA = '250424 Highway-Rail Grade Crossing Incident Data (Form 57).csv'
FN_DF_LABEL = 'df_news_label.csv'
FN_DICT_FORM57_PDF = 'form57_field_def_final.json'
FN_DICT_FORM57_CSV = 'form57_field_def_csv.json'
FN_PDF_FORM57 = 'FRA F 6180.57 (Form 57) form only.pdf'
FN_DF_OUTPUT = 'df_output.csv'
COLUMNS_CONTENT = ['np_url', 'tf_url', 'rd_url', 'gs_url', 'np_html', 'tf_html', 'rd_html', 'gs_html']
COLUMNS_LABEL = ['label_np_url', 'label_tf_url', 'label_rd_url', 'label_gs_url', 'label_np_html', 'label_tf_html', 'label_rd_html', 'label_gs_html']
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

path_df = DATA_FOLDER + FN_DF_DATA
path_df_label = DATA_FOLDER + FN_DF_LABEL
path_dict_form57_csv = DATA_FOLDER + FN_DICT_FORM57_CSV
path_pdf_form57 = DATA_FOLDER + FN_PDF_FORM57
path_df_output = DATA_FOLDER + FN_DF_OUTPUT

df_data = pd.read_csv(path_df)
df_data = df_data[df_data['State Name'] == 'CALIFORNIA']
df_data['hash_id'] = df_data.apply(utils_scrape.hash_row, axis=1)
df_data['Date'] = pd.to_datetime(df_data['Date'])
df_data = df_data[df_data['Date'] >= '2000-01-01']

SCRAPE_VERSION = 'rd_url'
df_label = pd.read_csv(path_df_label)
df_label = df_label[(df_label[COLUMNS_LABEL] == 1).any(axis=1)]
sr_content = df_label[SCRAPE_VERSION][df_label['label_' + SCRAPE_VERSION] == 1]

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

N_SIM = 1
MODEL_PATH = 'microsoft/Phi-4-mini-instruct'
# MODEL_PATH = 'nvidia/Llama-3.1-Nemotron-Nano-8B-v1'
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

config = dict_model[MODEL_PATH]
pipe = pipeline(model=MODEL_PATH, device_map=DEVICE, **config)
if pipe.generation_config.pad_token_id is None:
    pipe.generation_config.pad_token_id = pipe.tokenizer.eos_token_id
if pipe.generation_config.temperature == 0 or pipe.generation_config.do_sample == False:
    n_sim = 1
else:
    n_sim = N_SIM

question_base = (
        "From the context, find the information: "
    )
answer_format = (
    "The answer should be short and without explanation.\n"
    "If options are provided, answer only with codes.\n"
    "If you cannot find the relevant information, answer with 'Unknown'.\n"
)

############ extract information using csv-version json
with open(path_dict_form57_csv, 'r') as f:
    dict_form57_csv = json.load(f)

df_output = pd.DataFrame(sr_content)
list_col = list(map(lambda x: x['name'], dict_form57_csv.values()))
df_output.loc[:, list_col] = ''

QUESTION_BATCH = 'single' # single, category
for idx_content, content in tqdm(sr_content.items(), total=sr_content.size):
    
    if QUESTION_BATCH == 'single':
        for i, entry in tqdm(dict_form57_csv.items(), leave=False):
            name = entry.get('name')
            options = str(entry.get('choices', ''))
            
            question = question_base + name + options + '\n' + answer_format
            prompt = f"Context:\n{content}\n\nQuestion:\n{question}\n\nAnswer:\n"
            
            output = pipe(prompt)#, max_new_tokens=30)
            answer = output[0]['generated_text'].split('Answer:\n')[-1]

            df_output.loc[idx_content, name] = answer

    elif QUESTION_BATCH == 'category':
        for question_category, target_info in dict_target_info.items():
            
            for i, (field_num, column_name) in enumerate(target_info.items()):
                field = dict_form57_csv[str(field_num)]
                field_name = field['name']
                field_choice = field.get('choices', {})
                # if field_choice != {}:
                #     field_choice['0'] = 'Unknown'
                question_temp = f"{field_num}. {field_name}: {field_choice}"
                question += question_temp + "\n"
            
            question += answer_format
            first_key = list(target_info.keys())[0]
            prompt = f"Context:\n{content}\n\nQuestion:\n{question}\n\nAnswer:\n" + f"{first_key}."# {dict_form57_csv[str(first_key)]['name']}:"

            list_answer = []
            for _ in range(n_sim):
                start = time.time()
                output = pipe(prompt)
                end = time.time()
                end - start
                answers = output[0]['generated_text'].split('Answer:')[-1].strip()
                print(answers.split('\n'))
            pass
    
    if idx_content % 10 == 0:
        df_output.to_csv(path_df_output)

df_output.to_csv(path_df_output)