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
from modules.scrape import TableConfig, ScrapeNews, scrape_news
from modules.filter_news import check_article_realtime
from modules.populate_form import populate_fields
from modules.utils import prepare_df_record

args = {
    "c_api": "Google",
    "c_model": "gemini-2.5-flash",
    "c_n_generate": 4,
    "c_json_source": "img",
    "c_seed": 1,
    "r_api": "Huggingface",
    "r_model": "microsoft/phi-4",
    "r_n_generate": 1,
    "r_question_batch": "group"
}
cfg = build_config(args)
START_DATE = (datetime.datetime.today() - datetime.timedelta(days=2)).date().strftime("%Y-%m-%d")
END_DATE = (datetime.datetime.today() + datetime.timedelta(days=1)).date().strftime("%Y-%m-%d")
STATE = cfg.scrp.target_states[0]

st.set_page_config(layout="wide")


############### fetching recent news
config_df_record_news = TableConfig(cfg.path.df_record_news_realtime, ['query1', 'query2', 'county', 'state', 'city', 'highway', 'report_key', 'news_id'])
config_df_news_articles = TableConfig(cfg.path.df_news_articles_realtime, ['news_id', 'url', 'pub_date', 'title'] + list(cfg.scrp.news_crawlers))

# df_record = prepare_df_record(cfg)
# list_prior_info = ['Report Key', 'Railroad Name', 'Date', 'Nearest Station', 'County Name', 'State Name', 'City Name', 'Highway Name', 'Public/Private', 'Highway User', 'Equipment Type'] # keywords useful for searching
# df_record = df_record.sort_values(['County Name', 'Date'], ascending=[True, False])

scrape = ScrapeNews(config_df_record_news, config_df_news_articles)
scrape.load_df_record_news()
scrape.load_df_news_articles()

# list_query1 = ["train", "amtrak", "locomotive"]
# list_query2 = ["accident", "incident", "crash", "collide", "hit", "strike", "injure", "kill", "derail"]
list_query1 = ["train"]
list_query2 = ["accident"]

pbar_query1 = tqdm(list_query1, leave=False)
for query1 in pbar_query1:
    pbar_query1.set_description(query1)
    pbar_query2 = tqdm(list_query2, leave=False)
    for query2 in pbar_query2:
        pbar_query2.set_description(query2)
        
        if scrape.already_scraped():
            time.sleep(0.001)
            continue

        query = f'{query1} {query2} {STATE} after:{START_DATE} before:{END_DATE}'
        feed = scrape.get_RSS(query)
        assert feed['bozo'] == False
        
        scrape.load_driver()
        df_temp = scrape.get_article(feed)
        # scrape.append_df_record_news(df_temp)
        # scrape.save_df_record_news()
        scrape.quit_driver()

        if df_temp.shape[0] <= 1:
            time.sleep(7)

scrape.df_news_articles


############### filtering news
df_news_articles_realtime = scrape.df_news_articles

col_crawlers = list(cfg.scrp.news_crawlers)
col_crawlers_recent = [crawler + '_recent' for crawler in col_crawlers]
col_crawlers_state = [crawler + '_state' for crawler in col_crawlers]
if os.path.exists(cfg.path.df_news_articles_realtime_score):
    df_news_articles_score = pd.read_csv(cfg.path.df_news_articles_realtime_score)
else:
    df_news_articles = pd.read_csv(cfg.path.df_news_articles_realtime, parse_dates=['pub_date'])
    df_news_articles_score = df_news_articles.copy(deep=True)
    df_news_articles_score[col_crawlers_recent] = np.nan
    df_news_articles_score[col_crawlers_state] = np.nan

df_news_articles_score[col_crawlers] = df_news_articles_score[col_crawlers].apply(lambda col: col.str.strip())
mask_not_empty = df_news_articles_score[col_crawlers] != ''
df_news_articles_score[col_crawlers] = df_news_articles_score[col_crawlers].where(mask_not_empty)
df_news_articles_score = df_news_articles_score[df_news_articles_score[col_crawlers].any(axis=1)]


### remove articles not related to train accident
dict_answer_choice = {'YES': 1, 'NO': 0}

### remove articles out of recent date range
list_skip_date = df_news_articles_score[df_news_articles_score['pub_date'] < START_DATE].index.tolist()

### remove articles not related to recent train accident
question = f"Is this article reporting a train accident that occured this/last week? Answer only {'/'.join(dict_answer_choice)}"
df_news_articles_score = check_article_realtime(cfg, df_news_articles_score, list_skip_date, col_crawlers, col_crawlers_recent, dict_answer_choice, question, num_sim=1)

df_news_articles_score.to_csv(cfg.path.df_news_articles_realtime_score, index=False)

### remove articles not related to US STATE
mask_recent = df_news_articles_score[col_crawlers_recent] == 1
list_skip_recent = df_news_articles_score[~mask_recent.any(axis=1)].index.tolist()
question = f"Did the train accident occur in US {STATE}? Answer only {'/'.join(dict_answer_choice)}"
df_news_articles_score = check_article_realtime(cfg, df_news_articles_score, list_skip_recent, col_crawlers, col_crawlers_state, dict_answer_choice, question, num_sim=1)

df_news_articles_score.to_csv(cfg.path.df_news_articles_realtime_score, index=False)

mask_state = df_news_articles_score[col_crawlers_state] == 1

### filter news articles
df_news_articles_filter = df_news_articles_score.copy(deep=True)
df_news_articles_filter[col_crawlers] = df_news_articles_filter[col_crawlers].where(mask_recent.values & mask_state.values)
df_news_articles_filter = df_news_articles_filter[df_news_articles_filter[col_crawlers].any(axis=1)]

df_news_articles_filter['content'] = np.nan

### select the longest article among different crawlers
for idx, row in df_news_articles_filter.iterrows():
    content_max_len = max(row[col_crawlers].astype(str).values, key=len)
    df_news_articles_filter.loc[idx, 'content'] = content_max_len
df_news_articles_filter['content'].sample(1).values

df_news_articles_filter = df_news_articles_filter.drop(col_crawlers + col_crawlers_recent + col_crawlers_state, axis=1)
df_news_articles_filter.to_csv(cfg.path.df_news_articles_realtime_filter, index=False)

df_news_articles_filter


############### retrieval

# -------------------------------
# Load data
# -------------------------------

df_retrieval = utils.prepare_df_retrieval(cfg)
dict_col_indexing = utils.prepare_dict_col_indexing(cfg)
dict_idx_mapping, dict_idx_mapping_inverse = utils.prepare_dict_idx_mapping(cfg)
dict_form57 = utils.prepare_dict_form57(cfg)
dict_form57_group = utils.prepare_dict_form57_group(cfg)

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
