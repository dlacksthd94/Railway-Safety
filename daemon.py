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

START_DATE = (datetime.datetime.today() - datetime.timedelta(days=7)).date().strftime("%Y-%m-%d")
END_DATE = (datetime.datetime.today() + datetime.timedelta(days=1)).date().strftime("%Y-%m-%d")
STATE = cfg.scrp.target_states[0]


############### fetching recent news
df_record_news_realtime, df_news_articles_realtime = scrape_news_realtime(cfg, START_DATE, END_DATE, STATE)
print(f"############### Fetched news articles!! ###############")


############### filtering news
df_news_articles_realtime_filter = filter_news_realtime(cfg, START_DATE, STATE, END_DATE)
print(f"############### Filtered news articles!! ###############")


############### retrieval
df_retrieval_realtime = extract_keywords_realtime(cfg)
print(f"############### Extracted keywords!! ###############")
