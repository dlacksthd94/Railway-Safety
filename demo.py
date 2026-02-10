import json
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import pandas as pd
import base64
import io
import datetime
import time
import os
import numpy as np
from tqdm import tqdm

from modules import build_config, extract_keywords
from modules.scrape import scrape_news_realtime
from modules.filter_news import filter_news_realtime
from modules.populate_form import populate_fields
from modules.extract_keywords import extract_keywords_realtime
from modules.utils import prepare_dict_col_indexing, prepare_dict_form57_group, prepare_dict_form57_group, prepare_dict_idx_mapping, prepare_dict_form57

args = {
    "c_api": "Google",
    "c_model": "gemini-2.5-flash",
    "c_n_generate": 4,
    "c_json_source": "img",
    "c_seed": 1,
    "r_api": "Google",
    "r_model": "gemini-2.5-flash",
    "r_n_generate": 1,
    "r_question_batch": "group"
}
cfg = build_config(args)
START_DATE = (datetime.datetime.today() - datetime.timedelta(days=2)).date().strftime("%Y-%m-%d")
END_DATE = (datetime.datetime.today() + datetime.timedelta(days=1)).date().strftime("%Y-%m-%d")
STATE = cfg.scrp.target_states[0]

st.set_page_config(layout="wide")


############### fetching recent news
df_record_news_realtime, df_news_articles_realtime = scrape_news_realtime(cfg, START_DATE, END_DATE, STATE)


############### filtering news
df_news_articles_realtime_filter = filter_news_realtime(cfg, START_DATE, STATE, END_DATE)


############### retrieval
df_retrieval = extract_keywords_realtime(cfg)


# -------------------------------
# Load data
# -------------------------------

dict_col_indexing = prepare_dict_col_indexing(cfg)
dict_idx_mapping, dict_idx_mapping_inverse = prepare_dict_idx_mapping(cfg)
dict_form57 = prepare_dict_form57(cfg)
dict_form57_group = prepare_dict_form57_group(cfg)

dict_group_field_cols = {}
for name, group in dict_form57_group.items():
    dict_field_cols_temp = {}
    content_idx = df_retrieval.columns.get_loc('content')
    for field in group:
        cols_temp = []
        for col in df_retrieval.columns[content_idx + 1:]: # type: ignore
            if field == col.split('_')[0]:
                cols_temp.append(col)
        dict_field_cols_temp[field] = cols_temp
    dict_group_field_cols[name] = dict_field_cols_temp

st.session_state["news_df"] = df_retrieval

df = st.session_state.get("news_df", None)
if df is None:
    st.error("Please load your DataFrame into st.session_state['news_df']")
    st.stop()

# -------------------------------
# Sidebar — Date Range Filter
# -------------------------------
st.sidebar.header("Filters")

min_date = df['pub_date'].min().date()
max_date = df['pub_date'].max().date() + pd.Timedelta(days=1)

date_range = st.sidebar.date_input(
    "News Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

start_date, end_date = date_range if isinstance(date_range, tuple) else (min_date, max_date) # type: ignore
filtered_df = df[(df['pub_date'] >= pd.Timestamp(start_date)) & (df['pub_date'] <= pd.Timestamp(end_date))]

# -------------------------------
# PAGE LAYOUT
# -------------------------------
col_left, col_right = st.columns([1, 1])   # Left wider than right

with col_left:
    upper_left = st.container()
    lower_left = st.container()

with col_right:
    right_panel = st.container()

# Remember selection
if "selected_news_id" not in st.session_state:
    st.session_state.selected_news_id = None


# -------------------------------
# UPPER LEFT — News List
# -------------------------------
with upper_left:
    options = {
        f"{row.pub_date.date()} — {row.title[:80]}...": row.news_id
        for _, row in filtered_df.iterrows()
    }

    if options:
        selected_label = st.selectbox(
            "Select a news article:",
            list(options.keys()),
            index=0
        )
        st.session_state.selected_news_id = options[selected_label]
    else:
        st.info("No news found in selected date range.")


# -------------------------------
# Get selected news item
# -------------------------------
selected = None
if st.session_state.selected_news_id is not None:
    selected = df[df["news_id"] == st.session_state.selected_news_id].iloc[0]


# -------------------------------
# LOWER LEFT — Embedded Original Page
# -------------------------------

# Yes, you can embed a PIL image in Streamlit using st.image()
# Example:
if selected is not None:
    idx_content = selected.index.get_loc('content')
    sr_retrieved_info = selected.iloc[idx_content + 1:] # type: ignore
    draw = populate_fields(cfg, sr_retrieved_info)
    
    pil_image = draw._image
    st.image(pil_image, caption="PIL Image", use_container_width=False)
