import os
import pandas as pd
from config import build_config
from utils import make_dir
from scrape_news import scrape_news
from filter_news import filter_news
from convert_report_to_json import convert_to_json
from extract_keywords import extract_keywords
from utils import merge_df, get_acc_table, get_cov_table

print('###########################################################################')
print('###########################################################################')

############### config
cfg = build_config()
print(cfg.path.dir_conversion)
print(cfg.path.dir_retrieval)
print('------------Configuration DONE!!------------')


# ############### news scraping
# df_record_news, df_news_articles = scrape_news(cfg)
# print('------------Scraping DONE!!------------')


# # ############### news filtering
# df_news_articles_filter = filter_news(cfg)
# print('------------Filtering DONE!!------------')


# ############### convert to json
# dict_form57, dict_form57_group = convert_to_json(cfg)
# print('------------Conversion DONE!!------------')


# ############### extract keywords
# df_retrieval = extract_keywords(cfg)
# print('------------Retrieval DONE!!------------')


############### match samples manually by running following command in terminal (ONLY ONE-TIME TASK)
# streamlit run match_record_news.py
assert os.path.exists(cfg.path.df_match)
df_match = pd.read_csv(cfg.path.df_match)
df_match = df_match[df_match['match'] == 1]
print('------------Matching DONE!!------------')


############### annotate samples manually by running following command in terminal (ONLY ONE-TIME TASK)
# streamlit run annotate_news.py
assert os.path.exists(cfg.path.df_annotate)
df_annotate = pd.read_csv(cfg.path.df_annotate)
df_annotate = df_annotate[df_annotate['annotated'] == 1]
print('------------Annotating DONE!!------------')


# ############### merge news-record pair and retrieval results
df_merge = merge_df(cfg)
print('------------Merging DONE!!------------')


############### calculate the accuracy and coverage
assert os.path.exists(cfg.path.dict_idx_mapping), "Must map index names shared accross the models with form transcription manually"

# list_answer_type_selected = ['choice']
# df_acc = get_acc_table(path_df_record, path_dict_col_indexing, path_dict_idx_mapping, path_dict_answer_places, df_merge, list_answer_type_selected, config)
# idx_news_content = df_acc.columns.get_loc('content')
# acc = df_acc.iloc[:, idx_news_content + 1:].dropna(axis=1, how='all').mean().mean()
# print('ACCURACY\ndigit + text + choice:\t', acc)

# list_answer_type_selected = ['digit']
# df_acc = get_acc_table(path_df_record, path_dict_col_indexing, path_dict_idx_mapping, path_dict_answer_places, df_merge, list_answer_type_selected, config)
# idx_news_content = df_acc.columns.get_loc('content')
# acc = df_acc.iloc[:, idx_news_content + 1:].dropna(axis=1, how='all').mean().mean()
# print('ACCURACY\ndigit + text + choice:\t', acc)

# list_answer_type_selected = ['text']
# df_acc = get_acc_table(path_df_record, path_dict_col_indexing, path_dict_idx_mapping, path_dict_answer_places, df_merge, list_answer_type_selected, config)
# idx_news_content = df_acc.columns.get_loc('content')
# acc = df_acc.iloc[:, idx_news_content + 1:].dropna(axis=1, how='all').mean().mean()
# print('ACCURACY\ndigit + text + choice:\t', acc)

list_answer_type_selected = ['digit', 'text', 'choice']
df_acc = get_acc_table(df_merge, list_answer_type_selected, cfg)
idx_news_content = df_acc.columns.get_loc('content')
acc = df_acc.iloc[:, idx_news_content + 1:].dropna(axis=1, how='all').mean().mean()
print('ACCURACY\ndigit + text + choice:\t', acc)

list_answer_type_selected = ['digit', 'text', 'choice']
df_cov = get_cov_table(df_annotate, list_answer_type_selected, cfg)
idx_news_content = df_cov.columns.get_loc('content')
cov = df_cov.iloc[:, idx_news_content + 1:].dropna(axis=1, how='all').mean().mean()
print('COVERAGE\ndigit + text + choice:\t', cov)

print('------------Metrics DONE!!------------')

print('###########################################################################')
print('###########################################################################')
