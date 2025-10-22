import pandas as pd
import numpy as np
import os
from pprint import pprint
import streamlit as st
import pandas as pd
import os
import json
import json5

st.set_page_config(layout='wide')

def random_idx():
    df_candidate = st.session_state.df_annotate[st.session_state.df_annotate['annotated'].isna()]
    if df_candidate.shape[0] == 0:
        return None
    else:
        idx = df_candidate.sample(1).index.item()
        return idx

DIR_DATA_ROOT = 'data'
DIR_JSON = os.path.join(DIR_DATA_ROOT, 'json')
# DIR_CONV_MODEL = os.path.join(DIR_JSON, 'img_OpenAI_o4-mini_4')
DIR_CONV_MODEL = os.path.join(DIR_JSON, 'None_None_None_0')
DIR_RESULT = os.path.join(DIR_CONV_MODEL, 'result')
DIR_RETR_MODEL = os.path.join(DIR_RESULT, 'OpenAI_o4-mini_1_all')

path_form57_csv = os.path.join(DIR_DATA_ROOT, '250821 Highway-Rail Grade Crossing Incident Data (Form 57).csv')
path_form57_img = os.path.join(DIR_DATA_ROOT, 'FRA F 6180.57 (Form 57) form only.jpg')
path_df_record_news = os.path.join(DIR_DATA_ROOT, 'df_record_news.csv')
path_df_match = os.path.join(DIR_DATA_ROOT, 'df_match.csv')
path_df_annotate = os.path.join(DIR_DATA_ROOT, 'df_annotate.csv')
path_dict_col_indexing = os.path.join(DIR_DATA_ROOT, 'dict_col_indexing.jsonc')

path_form57_json = os.path.join(DIR_CONV_MODEL, 'form57.json')
path_dict_answer_places = os.path.join(DIR_CONV_MODEL, 'dict_answer_places.jsonc')
path_dict_idx_mapping = os.path.join(DIR_CONV_MODEL, 'dict_idx_mapping.jsonc')

path_df_form57_retrieval = os.path.join(DIR_RETR_MODEL, 'df_form57_retrieval.csv')

with open(path_dict_col_indexing, 'r') as f:
    dict_col_indexing = json5.load(f)

with open(path_dict_answer_places, 'r') as f:
    dict_answer_places = json5.load(f)

with open(path_dict_idx_mapping, 'r') as f:
    dict_idx_mapping = json5.load(f)
    dict_idx_mapping_inverse = {v: k for k, v in dict_idx_mapping.items()}

with open(path_form57_json, 'r') as f:
    dict_form57 = json.load(f)

df_retrieval = pd.read_csv(path_df_form57_retrieval)
df_retrieval = df_retrieval.rename(columns=dict_idx_mapping_inverse)
df_retrieval = df_retrieval.rename(columns={'20c_measure': '20c'})
df_retrieval = df_retrieval.drop(columns=['url', 'pub_date', 'title', 'content'])

if "df" not in st.session_state:
    if os.path.exists(path_df_annotate):
        df_annotate = pd.read_csv(path_df_annotate)
        st.session_state.df_annotate = df_annotate
    else:
        df_match = pd.read_csv(path_df_match)
        df_match = df_match[df_match['match'] == 1]
        df_annotate = df_match.copy(deep=True)
        idx_content = df_annotate.columns.get_loc('content')
        df_annotate = df_annotate.iloc[:, :idx_content + 1] # type: ignore
        df_annotate = df_annotate.drop(columns=['query1', 'query2', 'county', 'state', 'city', 'highway', 'title', 'pub_date', 'url'])
        df_annotate = df_annotate.merge(df_retrieval, on='news_id')
        
        idx_col_content = df_annotate.columns.get_loc('content')
        for col in df_annotate.columns[idx_col_content + 1:]: # type: ignore
            if col not in dict_col_indexing:
                df_annotate = df_annotate.drop(columns=[col])
        
        df_annotate['annotated'] = np.nan
        df_annotate.to_csv(path_df_annotate, index=False)
        st.session_state.df_annotate = df_annotate

if 'idx' not in st.session_state:
    st.session_state.idx = random_idx()

if st.session_state.idx:
    sr_annotate = st.session_state.df_annotate.iloc[st.session_state.idx, :]
    sr_annotate.name = 'retrieval'
    news_content = sr_annotate['content']

    col_news, col_df = st.columns(2)
    with col_news:
        st.write(f"# {st.session_state.df_annotate['annotated'].notna().sum()}/{st.session_state.df_annotate.shape[0]} samples have been annotated.")

        st.write(news_content)
    with col_df:
        df_check = pd.DataFrame(sr_annotate)
        df_check['name'] = df_check.rename(index=dict_col_indexing).index
        df_check['checkbox'] = False
        # mask_empty = (df_check[0].str.lower() == 'unknown') | df_check[0].isna() | (df_check[0] == '')
        # df_check['checkbox'] = ~mask_empty
        # df_check.loc[['news_id', 'report_key', '54'], 'checkbox'] = False
        df_check.loc['annotated', 'checkbox'] = True
        df_edited = st.data_editor(
            df_check,
            key="editor",
            use_container_width=True,
            hide_index=False,
            column_config={
                "retrieval": st.column_config.TextColumn(required=True),
                "name": st.column_config.TextColumn(required=True),
                "checkbox": st.column_config.CheckboxColumn(),
            },
            height=750,
        )
        # df_edited = st.data_editor(df_check, height=800)

        if st.button("Next", type="primary", help="Commit the edits and save to disk"):
            st.session_state.df_annotate.iloc[st.session_state.idx, 3:] = df_edited['checkbox'][3:]

            st.session_state.df_annotate.to_csv(path_df_annotate, index=False)

            st.session_state.idx = random_idx()
            st.rerun()
else:
    st.warning('All samples have been annotated!')

# df = pd.read_csv(path_form57_csv)
# pprint(df[df['Report Key'] == 'SCRT03082023202303'].to_dict(), indent=4, sort_dicts=False)