import pandas as pd
import os
import ast
import newspaper
from tqdm import tqdm

DATA_FOLDER = 'data/'

if os.path.exists(DATA_FOLDER + 'df_url_state.csv'):
    df_url_state = pd.read_csv(DATA_FOLDER + 'df_url_state.csv')

df_url_state['url'] = df_url_state['url'].apply(ast.literal_eval)
df_url_state = df_url_state.explode('url')
df_url_state = df_url_state[~df_url_state['url'].isna()]
df_url_state = df_url_state.drop_duplicates('url')
df_url_state = df_url_state.reset_index(drop=True)

for i, url in tqdm(df_url_state['url'].items(), total=df_url_state.shape[0]):
    try:
        article = newspaper.Article(url, language='en')
        article.download()
        article.parse()
        df_url_state.loc[i, 'title'] = article.title
        list_sentence = article.text.split('\n\n')
        list_content = []
        for sentence in list_sentence:
            if sentence != article.title and sentence != article.meta_description and sentence not in list_content:
                list_content.append(sentence)
        df_url_state.loc[i, 'content'] = '\n\n'.join(list_content)
        df_url_state.loc[i, 'keywords'] = '/'.join(article.tags)
        # df_url_state.loc[i, 'img_url'] = article.meta_img
        # df_url_state.loc[i, 'img_desc'] = article.meta_description
    except:
        pass

df_url_state = df_url_state.drop_duplicates(['content', 'title'])
df_url_state = df_url_state[df_url_state['content'] != '']
df_news_final = df_url_state.reset_index(drop=True)

fn = DATA_FOLDER + 'df_news_state.csv'
df_news_final.to_csv(fn, index=False)
df_news_final = pd.read_csv(fn)