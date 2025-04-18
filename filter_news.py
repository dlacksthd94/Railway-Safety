import pandas as pd
import json
import os
pd.set_option('display.max_rows', 100)

DATA_FOLDER = 'data/'
FN_DF = 'df_news.csv'
FN_DF_FILTER = 'df_news_filter.csv'

path_df = DATA_FOLDER + FN_DF
df = pd.read_csv(path_df, parse_dates=['pub_date'])

######### drop na
df = df.dropna(subset=['url'])
df = df.dropna(subset=['content'])

######### drop duplicates
df = df.drop_duplicates('url')

######### drop content not containing query1
path_json_list_keywords = DATA_FOLDER + 'dict_list_keywords.json'
with open(path_json_list_keywords, "r") as file:
    dict_list_keywords = json.load(file)
list_query1 = dict_list_keywords['list_query1']
pattern = '|'.join(list_query1)
contain_query1 = df['content'].str.contains(pattern, case=False, na=False) | df['title'].str.contains(pattern, case=False, na=False)

df = df[(contain_query1 | exclude)]

df.loc[exclude[exclude].index]

df = df.reset_index(drop=True)
path_df_filter = DATA_FOLDER + FN_DF_FILTER
df.to_csv(path_df_filter, index=False)