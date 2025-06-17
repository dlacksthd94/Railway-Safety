import pandas as pd
import json
import utils_scrape
from pprint import pprint

DATA_FOLDER = 'data/'
FN_DF_DATA = '250424 Highway-Rail Grade Crossing Incident Data (Form 57).csv'
FN_DICT_FORM57_CSV = 'form57_field_def_csv.json'

path_df = DATA_FOLDER + FN_DF_DATA
path_dict_form57_csv = DATA_FOLDER + FN_DICT_FORM57_CSV

df_data = pd.read_csv(path_df)
# df_data = df_data[df_data['State Name'] == 'CALIFORNIA']
df_data['hash_id'] = df_data.apply(utils_scrape.hash_row, axis=1)
df_data['Date'] = pd.to_datetime(df_data['Date'])
df_data = df_data[df_data['Date'] >= '2000-01-01']

list_col = df_data.columns.tolist()

############ Match the columns containing codes with the columns containing labels.
dict_code_label = {}
for i, col in enumerate(list_col):
    if col.endswith('Code'):
        col_code = col
        col_label = list_col[i + 1]
        dict_code_label[col_code] = col_label

############ Extract the options (multiple choices), if any
dict_entry_choice = {}
for col_code, col_label in dict_code_label.items():
    df_drop_na = df_data[[col_code, col_label]].dropna()
    df_astype = df_drop_na
    df_astype[col_code] = df_astype[col_code].apply(lambda x: int(x) if isinstance(x, float) else x)
    df_astype[col_code] = df_astype[col_code].astype(str)
    df_drop_dup = df_astype.drop_duplicates()
    df_sort = df_drop_dup.sort_values(col_code)
    
    dict_entry_choice_temp = df_sort.set_index(col_code)[col_label].to_dict()
    if len(dict_entry_choice_temp) <= 15:
        dict_entry_choice[col_label] = dict_entry_choice_temp
pprint(dict_entry_choice)

############ Convert into JSON
list_col_wo_code = list(filter(lambda col: not col.endswith('Code'), list_col))
list_col_wo_code.remove('hash_id')
dict_form57_csv = {i: {'name': col} for i, col in enumerate(list_col_wo_code)}
for i, dict_meta_info in dict_form57_csv.items():
    col = dict_meta_info['name']
    if col in dict_entry_choice:
        choices = dict_entry_choice[col]
        dict_form57_csv[i]['choices'] = choices
pprint(dict_form57_csv)

with open(path_dict_form57_csv, 'w') as f:
    json.dump(dict_form57_csv, f, indent=4)
