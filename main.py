import os
import argparse
import pandas as pd
import utils_scrape
import json5
import numpy as np
import tqdm

print('###########################################################################')
print('###########################################################################')

parser = argparse.ArgumentParser(
    description="Run main.py"
)

# Positional arguments (order matters)
# parser.add_argument("file", help="Path to input file")
# parser.add_argument("count", type=int, help="Number of iterations")

# Optional (flag-style) arguments
parser.add_argument(
    "--c_api",
    type=str,
    choices=['Huggingface', 'OpenAI', 'None'],
    # default="None",
    # default="Huggingface",
    default='OpenAI',
    help="API to use for processing"
)

parser.add_argument(
    "--c_model",
    type=str,
    # choices=['Qwen/Qwen2.5-VL-7B-Instruct', 'OpenGVLab/InternVL3-8B-hf', 'ByteDance-Seed/UI-TARS-1.5-7B', 'None'],
    # default='None',
    # default='OpenGVLab/InternVL3_5-38B-HF',
    # default='Qwen/Qwen2.5-VL-72B-Instruct',
    # default='llava-hf/llava-v1.6-34b-hf',
    default='o4-mini', # o1, o3, gpt-4.1 slightly poor (error in choice lists) / gpt-5 poor (too many errors) / gpt-4o very poor (not even completed)
    # default='o3-pro',
    help="Model to use for processing"
)

parser.add_argument(
    "--c_n_generate",
    type=int,
    # default=0,
    # default=1,
    default=4,
    # action="store_true",
    help="Number of generations"
)

parser.add_argument(
    "--c_json_source",
    type=str,
    choices=['csv', 'pdf', 'img'],
    # default='csv',
    default='img',
    help="Source of JSON data"
)

parser.add_argument(
    "--r_question_batch",
    type=str,
    choices=['one_pass', 'single', 'group'],
    # default='single',
    default='group',
    help="Batching strategy for questions"
)

args = parser.parse_args()

############### config
from config import Config, ConversionConfig, RetrievalConfig

config_conversion = ConversionConfig(api=args.c_api, model=args.c_model, n_generate=args.c_n_generate, json_source=args.c_json_source)
# config_retrieval = RetrievalConfig(api='Huggingface', model='microsoft/Phi-4-mini-instruct', n_generate=1, question_batch=args.r_question_batch) # bad
config_retrieval = RetrievalConfig(api='Huggingface', model='microsoft/phi-4', n_generate=1, question_batch=args.r_question_batch)
# "Qwen/Qwen3-4B ~ 32B" and "openai/gpt-oss-20b" take too long and too verbose

config = Config(conversion=config_conversion, retrieval=config_retrieval)

print('------------Configuration DONE!!------------')

############### paths
from utils import make_dir

DIR_DATA_ROOT = 'data'
DIR_DATA_JSON = 'json'
DIR_DATA_RESULT = 'result'

FN_DF_RECORD = '250821 Highway-Rail Grade Crossing Incident Data (Form 57).csv'

LIST_CRAWLER = ['np_url', 'tf_url', 'rd_url', 'gs_url', 'np_html', 'tf_html', 'rd_html', 'gs_html']
FN_DF_RECORD_NEWS = 'df_record_news.csv'
FN_DF_NEWS_ARTICLES = 'df_news_articles.csv'
FN_DF_NEWS_ARTICLES_SCORE = 'df_news_articles_score.csv'
FN_DF_NEWS_ARTICLES_FILTER = 'df_news_articles_filter.csv'

FN_DF_NEWS_LABEL = 'df_news_label.csv'

# NM_FORM57 = 'FRA F 6180.57 (Form 57) form only'
NM_FORM57 = 'FRA F 6180.57 (Form 57) form only'
FN_FORM57_PDF = f'{NM_FORM57}.pdf'
FN_FORM57_IMG = f'{NM_FORM57}.jpg'

FN_FORM57_JSON = f'form57.json'
FN_FORM57_JSON_GROUP = f'form57_group.json'

FN_DF_FORM57_RETRIEVAL = f'df_form57_retrieval.csv'

FN_DF_MATCH = 'df_match.csv'
FN_DICT_ANSWER_PLACES = 'dict_answer_places.jsonc'
FN_DICT_IDX_MAPPING = 'dict_idx_mapping.jsonc'
FN_DICT_COL_INDEXING = 'dict_col_indexing.jsonc'

conversion_model_replaced = config.conversion.model.replace('/', '@')
retrieval_model_replaced = config.retrieval.model.replace('/', '@')
dir_config_json = f'{config.conversion.json_source}_{config.conversion.api}_{conversion_model_replaced}_{config.conversion.n_generate}'
path_dir_config_json = os.path.join(DIR_DATA_ROOT, DIR_DATA_JSON, dir_config_json)
make_dir(path_dir_config_json)
dir_config_result = f'{config.retrieval.api}_{retrieval_model_replaced}_{config.retrieval.n_generate}_{config.retrieval.question_batch}'
path_dir_config_result = os.path.join(path_dir_config_json, DIR_DATA_RESULT, dir_config_result)
make_dir(path_dir_config_result)

path_df_record = os.path.join(DIR_DATA_ROOT, FN_DF_RECORD)

path_df_record_news = os.path.join(DIR_DATA_ROOT, FN_DF_RECORD_NEWS)
path_df_news_articles = os.path.join(DIR_DATA_ROOT, FN_DF_NEWS_ARTICLES)
path_df_news_articles_score = os.path.join(DIR_DATA_ROOT, FN_DF_NEWS_ARTICLES_SCORE)
path_df_news_articles_filter = os.path.join(DIR_DATA_ROOT, FN_DF_NEWS_ARTICLES_FILTER)

path_df_news_label = os.path.join(DIR_DATA_ROOT, FN_DF_NEWS_LABEL)
path_form57_pdf = os.path.join(DIR_DATA_ROOT, FN_FORM57_PDF)
path_form57_img = os.path.join(DIR_DATA_ROOT, FN_FORM57_IMG)
path_form57_json = os.path.join(path_dir_config_json, FN_FORM57_JSON)
path_form57_json_group = os.path.join(path_dir_config_json, FN_FORM57_JSON_GROUP)
path_df_form57_retrieval = os.path.join(path_dir_config_result, FN_DF_FORM57_RETRIEVAL)

path_df_match = os.path.join(DIR_DATA_ROOT, FN_DF_MATCH)
path_dict_answer_places = os.path.join(path_dir_config_json, FN_DICT_ANSWER_PLACES)
path_dict_idx_mapping = os.path.join(path_dir_config_json, FN_DICT_IDX_MAPPING)
path_dict_col_indexing = os.path.join(DIR_DATA_ROOT, FN_DICT_COL_INDEXING)

print(path_dir_config_json)
print(path_dir_config_result)

print('------------Setting path DONE!!------------')

# ###############
# from scrape_news import scrape_news

# df_record_news, df_news_articles = scrape_news(path_df_record_news, path_df_news_articles, path_df_record, LIST_CRAWLER)

# ############### rule-based news filtering
# from filter_news import filter_news

# df_news_articles_filter = filter_news(path_df_news_articles, path_df_news_articles_score, path_df_news_articles_filter, LIST_CRAWLER)

############### convert to json
from convert_report_to_json import csv_to_json, pdf_to_json, img_to_json

if config.conversion.json_source == 'csv':
    dict_form57 = csv_to_json(path_df_record, path_form57_json)
elif config.conversion.json_source == 'pdf':
    dict_form57, dict_form57_group = pdf_to_json(path_form57_pdf, path_form57_json, path_form57_json_group, config.conversion)
elif config.conversion.json_source == 'img':
    dict_form57, dict_form57_group = img_to_json(path_form57_img, path_form57_json, path_form57_json_group, config.conversion)

print('------------Conversion DONE!!------------')

############### extract keywords
from extract_keywords import extract_keywords

df_retrieval = extract_keywords(path_form57_json, path_form57_json_group, path_df_form57_retrieval, path_df_news_articles_filter, path_df_match, path_dict_answer_places, config)

print('------------Retrieval DONE!!------------')

############### match samples manually via `match_record_news.py` (ONLY ONE-TIME TASK)
assert os.path.exists(path_df_match)
df_match = pd.read_csv(path_df_match)
df_match = df_match[df_match['match'] == 1]

print('------------Matching DONE!!------------')

# ############### merge news-record pair and retrieval results
assert df_match['news_id'].is_unique and df_retrieval['news_id'].is_unique, '==========Warning: News is not unique!!!==========='

idx_content_match = df_match.columns.get_loc('content')
df_match = df_match.iloc[:, :idx_content_match + 1]

df_retrieval_drop = df_retrieval.set_index('news_id')
idx_content_retrieval = df_retrieval_drop.columns.get_loc('content')
df_retrieval_drop = df_retrieval_drop.iloc[:, idx_content_retrieval + 1:]

df_merge = df_match.merge(df_retrieval_drop, left_on='news_id', right_index=True, how='inner')

print('------------Merging DONE!!------------')

############### calculate the accuracy
from utils import get_acc_table

assert os.path.exists(path_dict_idx_mapping), "Must map index names shared accross the models with form transcription manually"

list_answer_type_selected = ['digit', 'text', 'choice']
df_acc = get_acc_table(path_df_record, path_dict_col_indexing, path_dict_idx_mapping, path_dict_answer_places, df_merge, list_answer_type_selected, config)
acc = df_acc.loc[:, '1':].dropna(axis=1, how='all').mean().mean()
print('digit + text + choice:\t', acc)

# list_answer_type_selected = ['digit']
# df_acc = get_acc_table(path_df_record, path_dict_col_indexing, path_dict_idx_mapping, path_dict_answer_places, df_merge, list_answer_type_selected, config)
# acc = df_acc.loc[:, '1':].dropna(axis=1, how='all').mean().mean()
# print('digit:\t', acc)

# list_answer_type_selected = ['choice']
# df_acc = get_acc_table(path_df_record, path_dict_col_indexing, path_dict_idx_mapping, path_dict_answer_places, df_merge, list_answer_type_selected, config)
# acc = df_acc.loc[:, '1':].dropna(axis=1, how='all').mean().mean()
# print('choice:\t', acc)

# list_answer_type_selected = ['text']
# df_acc = get_acc_table(path_df_record, path_dict_col_indexing, path_dict_idx_mapping, path_dict_answer_places, df_merge, list_answer_type_selected, config)
# acc = df_acc.loc[:, '1':].dropna(axis=1, how='all').mean().mean()
# print('text:\t', acc)

print('------------accuracy DONE!!------------')