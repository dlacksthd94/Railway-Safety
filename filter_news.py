import pandas as pd
import json
import os
pd.set_option('display.max_rows', 100)

DATA_FOLDER = 'data/'
FN_DF = 'df_news.csv'
FN_DF_FILTER = 'df_news_filter.csv'
COLUMNS_CONTENT = ['np_url', 'tf_url', 'rd_url', 'gs_url', 'np_html', 'tf_html', 'rd_html', 'gs_html']

path_df = DATA_FOLDER + FN_DF
path_df_filter = DATA_FOLDER + FN_DF_FILTER

df = pd.read_csv(path_df, parse_dates=['pub_date'])

######### drop non-article
no_article_patterns = [
    '^\s+$',
    'Press & Hold to confirm you',
    'You are not authorized to access',
    'Edge: Too Many Requests',
    '403 Forbidden',
    'These cookies are necessary for our services to function',
    'Selling, Sharing, Targeted Advertising',
    'Please enable JS and disable any ad blocker',
    'We may use personal information to support',
    'Access to this site has been denied.',
    'This website is using a security service to protect itself from online attacks',
    'we provide online advertising services that use cookies and similar technologies to collect information',
    "We won't sell or share your personal information to inform the ads you see.",
    'we process personal information to inform which ads you see on our services',
]
pattern = r'|'.join(no_article_patterns)
df[COLUMNS_CONTENT] = df[COLUMNS_CONTENT].apply(lambda col: col.replace(to_replace=pattern, value=float('nan'), regex=True))

######### drop na
df = df.dropna(subset=['news_id'])
df = df.dropna(subset=COLUMNS_CONTENT, how='all')

######### save
df = df.reset_index(drop=True)
df.to_csv(path_df_filter, index=False)