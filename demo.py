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
from modules.utils import prepare_dict_col_indexing, prepare_dict_form57_group, prepare_dict_form57_group, prepare_dict_idx_mapping, prepare_dict_form57, prepare_df_retrieval_realtime


st.set_page_config(layout="wide")

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


# -------------------------------
# Load data
# -------------------------------
df_retrieval_realtime = prepare_df_retrieval_realtime(cfg)
df_retrieval_realtime = df_retrieval_realtime.sort_values(by='accident_date', ascending=True)

st.session_state["news_df"] = df_retrieval_realtime

df = st.session_state.get("news_df", None)
if df is None:
    st.error("Please load your DataFrame into st.session_state['news_df']")
    st.stop()

# -------------------------------
# Sidebar — Date Range Filter
# -------------------------------
st.sidebar.header("Filters")

today = datetime.datetime.today()
min_date = df['pub_date'].min().date()
max_date = today + pd.Timedelta(days=1)

date_range = st.sidebar.date_input(
    "News Date Range",
    value=(today - datetime.timedelta(days=7), today),
    min_value=min_date,
    max_value=max_date,
)

start_date, end_date = date_range if isinstance(date_range, tuple) else (min_date, max_date) # type: ignore
filtered_df = df[(df['pub_date'] >= pd.Timestamp(start_date)) & (df['pub_date'] <= pd.Timestamp(end_date))]


############### cleanse information

# overwrite '5_1', '5_2', '5_3' with accident date
filtered_df.loc[:, '5_1'] = filtered_df['accident_date'].dt.month.astype(str)
filtered_df.loc[:, '5_2'] = filtered_df['accident_date'].dt.day.astype(str)
filtered_df.loc[:, '5_3'] = filtered_df['accident_date'].dt.year.astype(str)

filtered_df.loc[:, '6_1'] = filtered_df['accident_date'].dt.strftime('%I:%M')
filtered_df.loc[:, '6_2'] = filtered_df['accident_date'].dt.strftime('%p')

filtered_df['9'] = filtered_df['9'].str.replace('county', '', case=False).str.strip()
filtered_df['11'] = filtered_df['11'].str.replace('city', '', case=False).str.strip()


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
        f"{row['accident_date'].date()} — {row.title[:80]}...": row.news_id
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
    selected = filtered_df[filtered_df["news_id"] == st.session_state.selected_news_id].iloc[0]


# -------------------------------
# LOWER LEFT — Populate form and show image
# -------------------------------

with lower_left:
    if selected is not None:
        idx_content = selected.index.get_loc('content')
        sr_retrieved_info = selected.iloc[idx_content + 1:] # type: ignore
        draw = populate_fields(cfg, sr_retrieved_info)
        
        pil_image = draw._image
        st.image(pil_image, caption="Populated Form Image", use_container_width=False)

        # download button for the image
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        st.download_button(
            label="Download Populated Form Image",
            data=img_byte_arr,
            file_name=f"populated_form_{selected.news_id}.png",
            mime="image/png"
        )
    else:
        st.info("Please select a news article to see the populated form.")
