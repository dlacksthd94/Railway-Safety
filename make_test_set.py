import pandas as pd
import numpy as np
import utils_scrape
import os
from pprint import pprint
import streamlit as st
import pandas as pd
import os
import json
import json5

st.set_page_config(layout='wide')

DIR_DATA_ROOT = 'data'

path_form57_csv = 'data/250424 Highway-Rail Grade Crossing Incident Data (Form 57).csv'
path_df_form57_retireval = 'data/json/pdf_OpenAI_o4-mini_8/result/Huggingface_microsoft/Phi-4-mini-instruct_1_group/df_form57_retrieval.csv'
path_df_test_set = os.path.join(DIR_DATA_ROOT, 'df_test_set.csv')
path_dict_col_idx_name = os.path.join(DIR_DATA_ROOT, 'dict_col_idx_name.jsonc')
def as_float(val):
    if isinstance(val, str) and val.isdigit():
        return float(val)
    else:
        return val

with open(path_dict_col_idx_name, 'r') as f:
    dict_col_idx_name = json5.load(f)

list_col_idx = ['1', '9', '11', '13', '38', '39']
list_col_name = [dict_col_idx_name[col] for col in list_col_idx]
list_col_name_additional = ['Time', 'Highway Name', 'Nearest Station', 'Crossing Users Killed', 'Crossing Users Injured', 'Passengers Killed', 'Passengers Injured', 'Equipment Type', 'Narrative']

list_pastel = ["#FFB3BA", "#BAE1FF"]

############ hand annotation
def load_df_csv():
    df_form57_csv = pd.read_csv(path_form57_csv)
    df_form57_csv = df_form57_csv[df_form57_csv['State Name'] == 'CALIFORNIA']
    df_form57_csv['hash_id'] = df_form57_csv.apply(utils_scrape.hash_row, axis=1)
    df_form57_csv = df_form57_csv.set_index('hash_id')
    df_form57_csv['Date'] = pd.to_datetime(df_form57_csv['Date'])

    st.session_state.df_form57_csv = df_form57_csv

def load_df_test_set():
    if os.path.exists(path_df_test_set):
        df_test_set = pd.read_csv(path_df_test_set)
        df_test_set['pub_date'] = pd.to_datetime(df_test_set['pub_date'])

    else:
        df_retrieval = pd.read_csv(path_df_form57_retireval)
        df_retrieval['pub_date'] = pd.to_datetime(df_retrieval['pub_date'])

        (~df_retrieval.isin(['Unknown', "'Unknown", '(Unknown)'])).sum().loc['1':].sort_values(ascending=False)[:30] # num of non-unknown values per column

        df_retrieval['9'] = df_retrieval['9'].str.upper().str.replace('COUNTY', '').str.strip()
        df_retrieval['11'] = df_retrieval['11'].str.upper().str.replace('', '').str.strip()
        df_retrieval['13'] = df_retrieval['13'].str.replace("'", '').replace('Auto', 'A')
        df_retrieval['38'] = df_retrieval['38'].str.replace(r'\D+', '', regex=True)
        df_retrieval['39'] = df_retrieval['39'].str.replace("'", '').map({'1': 'Male', '2': 'Female', 'Male': 'Male', 'Female': 'Female', '1 (Male)': 'Male', '2 (Female)': 'Female'})
        df_retrieval['44'] = df_retrieval['44'].str.replace("'", '').map({'1': '1', '2': '2', '3': '3', 'Killed': '1', 'Injured': '2', 'Uninjured': '3', '1 (Killed)': '1', '2: Injured': '2'})

        df_test_set = df_retrieval.copy(deep=True)
        df_test_set['match'] = np.nan

        for idx, sr_retireval in df_retrieval.iterrows():
            sr_retireval = sr_retireval[['report_key', 'pub_date'] + list_col_idx]
            report_key = sr_retireval.pop('report_key')
            sr_retireval = sr_retireval.apply(as_float)
            sr_form57_csv = df_form57_csv.loc[report_key]
            sr_form57_csv = sr_form57_csv[['Date'] + list_col_name]
            mask = sr_form57_csv.values == sr_retireval.values
            if not any(mask):
                df_test_set.loc[idx, 'match'] = 0

        df_test_set.to_csv(path_df_test_set, index=False)

    st.session_state.df_test_set = df_test_set
    st.session_state.idx = 0

def advance_and_mark(answer: str):
    """Mark current row as Yes/No, advance index by 1."""
    i = st.session_state.idx
    st.session_state.df_test_set.at[i, "match"] = answer
    st.session_state.idx += 1

if "df_csv" not in st.session_state:
    load_df_csv()

if "df_test_set" not in st.session_state:
    load_df_test_set()

if 'i_color' not in st.session_state:
    st.session_state.i_color = 0
    st.session_state.color = list_pastel[st.session_state.i_color]

df_test_set = st.session_state.df_test_set
df_form57_csv = st.session_state.df_form57_csv
idx = st.session_state.idx

# Skip already-reviewed rows
while idx < len(df_test_set) and pd.notna(df_test_set.at[idx, "match"]):
    idx += 1
    st.session_state.idx = idx

if idx < len(df_test_set):
    col_left, col_right = st.columns(2)

    with col_left:
        sr_match = df_test_set.loc[idx, ['report_key', 'pub_date'] + list_col_idx]
        report_key = sr_match.pop('report_key')
        if idx > 0:
            report_key_prev = df_test_set.loc[idx - 1, 'report_key']
        else:
            report_key_prev = None
        sr_match = sr_match.apply(as_float)
        sr_match = pd.concat([sr_match, pd.Series([np.nan] * len(list_col_name_additional))])
        sr_form57_csv = df_form57_csv.loc[report_key]
        sr_form57_csv = sr_form57_csv[['Date'] + list_col_name + list_col_name_additional]
        mask = sr_form57_csv.values == sr_match.values
        with st.form('### Comparison'):
            df_compare = pd.DataFrame({'csv': sr_form57_csv.values, 'news': sr_match.values, 'match': mask}, index=sr_form57_csv.index)
            day_name_csv, day_name_news = sr_form57_csv['Date'].day_name(), sr_match['pub_date'].day_name()
            df_compare.loc['day_name'] = [day_name_csv, day_name_news, day_name_csv == day_name_news]
            
            if report_key != report_key_prev:
                st.session_state.i_color += 1
                st.session_state.color = list_pastel[st.session_state.i_color % 2]
                df_test_set.to_csv(path_df_test_set, index=False)
            else:
                st.session_state.color = st.session_state.color
            df_compare_styled = df_compare.style.set_properties(**{"background-color": st.session_state.color}, subset=["csv"])
            st.dataframe(df_compare_styled, height=int(35.5 * (df_compare.shape[0] + 1)))
            
            _ = st.form_submit_button("Save changes", disabled=True)

        st.markdown(f"### [{idx}] Does this news match the record?")

        col_yes, col_no, col_unsure, col_including = st.columns(4)
        col_yes.button("Yes", on_click=advance_and_mark, args=(1,))
        col_no.button("No",  on_click=advance_and_mark, args=(0,))
        col_unsure.button("Unsure",  on_click=advance_and_mark, args=(2,))
        col_including.button("Including",  on_click=advance_and_mark, args=(3,))

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col2:
            if st.button("💾 Save updates"):
                st.session_state.df_test_set.to_csv(path_df_test_set, index=False)
                st.success(f"Saved to **{path_df_test_set}**!")
    
    with col_right:
        st.markdown("### News article")
        st.write(df_test_set.loc[idx, 'rd_url'])

else:
    st.success("✅ All news reviewed!")
