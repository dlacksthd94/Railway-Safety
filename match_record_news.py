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
from utils import as_float, lower_str

st.set_page_config(layout='wide')

DIR_DATA_ROOT = 'data'
DIR_JSON = os.path.join(DIR_DATA_ROOT, 'json')
DIR_CONV_MODEL = os.path.join(DIR_JSON, 'img_OpenAI_o4-mini_4')
DIR_RESULT = os.path.join(DIR_CONV_MODEL, 'result')
DIR_RETR_MODEL = os.path.join(DIR_RESULT, 'Huggingface_microsoft@phi-4_1_group')

path_form57_csv = os.path.join(DIR_DATA_ROOT, '250821 Highway-Rail Grade Crossing Incident Data (Form 57).csv')
# path_df_form57_retrieval = 'data/json/pdf_OpenAI_o4-mini_8/result/Huggingface_microsoft/Phi-4-mini-instruct_1_group/df_form57_retrieval.csv'
path_df_form57_retrieval = os.path.join(DIR_RETR_MODEL, 'df_form57_retrieval.csv')
path_df_record_news = os.path.join(DIR_DATA_ROOT, 'df_record_news.csv')
path_df_match = os.path.join(DIR_DATA_ROOT, 'df_match.csv')
path_dict_idx_mapping = os.path.join(DIR_CONV_MODEL, 'dict_idx_mapping.jsonc')
path_dict_col_indexing = os.path.join(DIR_DATA_ROOT, 'dict_col_indexing.jsonc')

with open(path_dict_col_indexing, 'r') as f:
    dict_col_indexing = json5.load(f)

with open(path_dict_idx_mapping, 'r') as f:
    dict_idx_mapping = json5.load(f)

list_col_idx_form57 = ['1', '6', '10', '9', '11', '7', '12', '13', '20a', '20b', '38', '39', '44', '46_killed', '46_injured', '48', '49_killed', '49_injured', '50', '52_killed', '52_injured', '54']
list_col_idx_json = [dict_idx_mapping[col_idx] for col_idx in list_col_idx_form57]
list_col_idx_json = ['NA' if idx == '' else idx for idx in list_col_idx_json]
list_col_name = [dict_col_indexing[col] for col in list_col_idx_form57]

list_pastel = ["#FFB3BA", "#BAE1FF"]

############ hand annotation
def load_df_csv():
    df_form57_csv = pd.read_csv(path_form57_csv)
    df_form57_csv = df_form57_csv[df_form57_csv['State Name'] == 'CALIFORNIA']
    df_form57_csv['Date'] = pd.to_datetime(df_form57_csv['Date'])
    df_form57_csv = df_form57_csv.set_index('Report Key')
    df_form57_csv['User Sex'] = df_form57_csv['User Sex'].map({'Male': 1, 'Female': 2})

    st.session_state.df_form57_csv = df_form57_csv

def load_df_retrieval():
    df_retrieval = pd.read_csv(path_df_form57_retrieval)
    df_retrieval['pub_date'] = pd.to_datetime(df_retrieval['pub_date'])
    return df_retrieval

def load_df_record_news():
    df_record_news = pd.read_csv(path_df_record_news)
    return df_record_news

def load_df_match():
    if os.path.exists(path_df_match):
        df_match = pd.read_csv(path_df_match)
        df_match['pub_date'] = pd.to_datetime(df_match['pub_date'])

    else:
        df_retrieval = load_df_retrieval()
        df_record_news = load_df_record_news()
        
        df_record_news = df_record_news.dropna(subset=['news_id'])
        df_record_news = df_record_news.merge(st.session_state.df_form57_csv['Date'], left_on='report_key', right_index=True, how='left')
        df_record_news = df_record_news.rename(columns={'Date': 'date'})
        
        df_match = df_record_news.merge(df_retrieval, on='news_id')
        df_match = df_match[(df_match['date'] - pd.Timedelta(days=1)) <= df_match['pub_date']]
        df_match = df_match.drop('date', axis=1)
        df_match['NA'] = ''

        # len(df_match['news_id'].unique())
        # len(df_match['report_key'].unique())
        # (~df_match.isin(['Unknown', "'Unknown", '(Unknown)'])).sum().iloc[12:].sort_values(ascending=False)[:30] # num of non-unknown values per column

        # df_match['9'] = df_match['9'].str.upper().str.replace('COUNTY', '').str.strip()
        # df_match['11'] = df_match['11'].str.upper().str.replace('', '').str.strip()
        # df_match['13'] = df_match['13'].str.replace("'", '').replace('Auto', 'A')
        # df_match['38'] = df_match['38'].str.replace(r'\D+', '', regex=True)
        df_match['39'] = df_match['39'].apply(lambda x: 2 if 'female' in x.lower() else 1 if 'male' in x.lower() else x)
        # df_match['44'] = df_match['44'].str.replace("'", '').map({'1': '1', '2': '2', '3': '3', 'Killed': '1', 'Injured': '2', 'Uninjured': '3', '1 (Killed)': '1', '2: Injured': '2'})
        
        df_match['match'] = np.nan
        
        # for idx, sr_match in df_match.iterrows():
        #     sr_match = sr_match[['report_key', 'pub_date'] + list_col_idx_json]
        #     report_key = sr_match.pop('report_key')
        #     sr_match = sr_match.apply(as_float)
        #     sr_match['pub_date'] = sr_match['pub_date'].date()
        #     sr_match = sr_match.apply(lambda s: s.lower() if isinstance(s, str) else s)
        #     sr_form57_csv = st.session_state.df_form57_csv[report_key].squeeze()
        #     sr_form57_csv = sr_form57_csv[['Date'] + list_col_name]
        #     sr_form57_csv['Date'] = sr_form57_csv['Date'].date()
        #     sr_form57_csv = sr_form57_csv.apply(lambda s: s.lower() if isinstance(s, str) else s)
        #     mask = sr_form57_csv.values == sr_match.values
        #     if not any(mask):
        #         df_match.loc[idx, 'match'] = 0
        # df_match['match'].isna().sum()

        df_match.to_csv(path_df_match, index=False)

    st.session_state.df_match = df_match
    st.session_state.idx = 0

def advance_and_mark(answer: str):
    """Mark current row as Yes/No, advance index by 1."""
    i = st.session_state.idx
    st.session_state.df_match.at[i, "match"] = answer
    if answer == 1:
        news_id = st.session_state.df_match.at[i, 'news_id']
        df_duplicated = st.session_state.df_match[st.session_state.df_match['news_id'] == news_id]
        idx_remove = df_duplicated[df_duplicated['match'] != 1].index
        st.session_state.df_match.loc[idx_remove, 'match'] = 0
    st.session_state.idx += 1

if "df_csv" not in st.session_state:
    load_df_csv()

if "df_match" not in st.session_state:
    load_df_match()

if 'i_color' not in st.session_state:
    st.session_state.i_color = 0
    st.session_state.color = list_pastel[st.session_state.i_color]

df_match = st.session_state.df_match
df_form57_csv = st.session_state.df_form57_csv
idx = st.session_state.idx

# Skip already-reviewed rows
while idx < len(df_match) and pd.notna(df_match.at[idx, "match"]):
    idx += 1
    st.session_state.idx = idx

if idx < len(df_match):
    col_left, col_right = st.columns(2)

    with col_left:
        sr_match = df_match.loc[idx, ['report_key', 'pub_date'] + list_col_idx_json]
        report_key = sr_match.pop('report_key')
        if idx > 0:
            report_key_prev = df_match.loc[idx - 1, 'report_key']
        else:
            report_key_prev = None
        sr_match = sr_match.apply(as_float)
        sr_form57_csv = df_form57_csv.loc[report_key]
        sr_form57_csv = sr_form57_csv[['Date'] + list_col_name]
        sr_form57_csv = sr_form57_csv.apply(as_float)
        mask = sr_form57_csv.apply(lower_str).values == sr_match.apply(lower_str).values
        sr_mask = pd.Series(mask, index=['Date'] + list_col_name)
        if (sr_match['pub_date'] - sr_form57_csv['Date']).days <= 1:
            sr_mask['Date'] = True
        if sr_match['1'] in sr_form57_csv['Railroad Name']:
            sr_mask['Railroad Name'] = True
        with st.form('### Comparison'):
            df_compare = pd.DataFrame({'csv': sr_form57_csv.values, 'news': sr_match.values, 'match': mask}, index=sr_form57_csv.index)
            day_name_csv, day_name_news = sr_form57_csv['Date'].day_name(), sr_match['pub_date'].day_name()
            df_compare.loc['day_name'] = [day_name_csv, day_name_news, day_name_csv == day_name_news]
            
            if report_key != report_key_prev:
                st.session_state.i_color += 1
                st.session_state.color = list_pastel[st.session_state.i_color % 2]
            else:
                st.session_state.color = st.session_state.color
            df_compare_styled = df_compare.style.set_properties(**{"background-color": st.session_state.color}, subset=["csv"])
            st.dataframe(df_compare_styled, height=int(35.5 * (df_compare.shape[0] + 1)))
            
            _ = st.form_submit_button("Save changes", disabled=True)

    with col_right:
        st.markdown(f"### [{idx}/{df_match.shape[0]}] Does this news match the record?")

        col_yes, col_no, col_unsure, col_including = st.columns(4)
        pub_date_0 = sr_match['pub_date'].strftime("%Y-%m-%d")
        pub_date_1 = (sr_match['pub_date'] - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        pub_date_2 = (sr_match['pub_date'] - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        pub_date_3 = (sr_match['pub_date'] - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        pub_date_4 = (sr_match['pub_date'] - pd.Timedelta(days=4)).strftime("%Y-%m-%d")

        st.write('-0 days:  ' + '|'.join(df_form57_csv[df_form57_csv['Date'] == pub_date_0]['City Name'].astype(str).to_list()))
        st.write('-1 days:  ' + '|'.join(df_form57_csv[df_form57_csv['Date'] == pub_date_1]['City Name'].astype(str).to_list()))
        st.write('-2 days:  ' + '|'.join(df_form57_csv[df_form57_csv['Date'] == pub_date_2]['City Name'].astype(str).to_list()))
        st.write('-3 days:  ' + '|'.join(df_form57_csv[df_form57_csv['Date'] == pub_date_3]['City Name'].astype(str).to_list()))
        st.write('-4 days:  ' + '|'.join(df_form57_csv[df_form57_csv['Date'] == pub_date_4]['City Name'].astype(str).to_list()))
        
        with col_yes:
            st.button("Yes", on_click=advance_and_mark, args=(1,))
        with col_no:
            st.button("No",  on_click=advance_and_mark, args=(0,))
        with col_unsure:
            st.button("Unsure",  on_click=advance_and_mark, args=(2,))
        with col_including:
            st.button("Including",  on_click=advance_and_mark, args=(3,))

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col2:
            if st.button("💾 Save updates"):
                st.session_state.df_match.to_csv(path_df_match, index=False)
                st.success(f"Saved to **{path_df_match}**!")

        st.markdown("### News article")
        st.write(df_match.loc[idx, 'url'])
        st.write(df_match.loc[idx, 'content'])

else:
    st.success("✅ All news reviewed!")

if __name__ == "__main__":
    # df_form57_csv[df_form57_csv['Highway Name'].str.contains('ENTERPRISE').fillna(False)]['Date']
    df_form57_csv[df_form57_csv['City Name'].str.contains('BURLINGAME').fillna(False)]['Date']
    df_form57_csv[df_form57_csv['Date'] == '2019-07-10']['City Name']
    # df_match[df_match['match'] == 1]