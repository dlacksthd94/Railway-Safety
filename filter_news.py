from transformers import pipeline
import pandas as pd
import numpy as np
import re
import utils
from tqdm import tqdm
import os
import torch
import gc
from itertools import combinations

pd.set_option('display.max_colwidth', 30)

# MODEL_PATH = 'microsoft/Phi-4-mini-instruct'
# MODEL_PATH = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
MODEL_PATH = 'Qwen/Qwen2.5-7B-Instruct-1M'
# DEVICE = 'cuda:1'
DEVICE = 'auto'

def text_binary_classification(pipe, prompt, dict_answer_choice, num_sim):
    list_output = pipe(prompt, max_new_tokens=1, num_return_sequences=num_sim, return_full_text=False)
    list_answer = list(map(lambda output: output['generated_text'].upper(), list_output))
    list_answer_filter = list(filter(lambda answer: answer in dict_answer_choice, list_answer))
    list_answer_map = list(map(lambda answer: dict_answer_choice[answer], list_answer_filter))
    return list_answer_map

def text_generation(pipe, prompt, max_new_tokens=4096):
    output = pipe(prompt, max_new_tokens=max_new_tokens, return_full_text=False)
    answer = output[0]['generated_text']
    return answer

def check_article(df_news_articles_score, list_skip, list_crawler, list_crawler_score, dict_answer_choice, question, num_sim=1):
    """
    classify each scraped article (per crawler) as reporting a train accident and store an averaged binary score.
    """
    assert len(set(dict_answer_choice.values())) == len(dict_answer_choice.values())
    pipe = pipeline(model=MODEL_PATH, device_map=DEVICE)

    for idx, row in tqdm(df_news_articles_score.iterrows(), total=df_news_articles_score.shape[0]):
        if idx in list_skip:
            continue
        if row[list_crawler_score].notna().any():
            continue
        sr_content = row[list_crawler]
        dict_column_score = {crawler_score: float('nan') for crawler_score in list_crawler_score}
        for col_crawler, col_score in tqdm(zip(list_crawler, list_crawler_score), total=len(list_crawler), leave=False):
            content = sr_content[col_crawler]
            if pd.isna(content):
                dict_column_score[col_score] = dict_answer_choice['NO']
            else:
                try:
                    prompt = f"Context:\n{content}\n\nQuestion:\n{question}\n\nAnswer:\n"
                    list_answer_map = text_binary_classification(pipe, prompt, dict_answer_choice, num_sim=num_sim)
                    if len(list_answer_map) > 0:
                        dict_column_score[col_score] = np.mean(list_answer_map)
                except:
                    pass
        df_news_articles_score.loc[idx, list_crawler_score] = dict_column_score.values()

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return df_news_articles_score

def filter_news(path_df_news_articles, path_df_news_articles_score, path_df_news_articles_filter, list_crawler):
    list_crawler_accd = [col + '_accd' for col in list_crawler]
    list_crawler_hwuser = [crawler + '_hwuser' for crawler in list_crawler]
    list_crawler_recent = [crawler + '_recent' for crawler in list_crawler]
    list_crawler_bp = [crawler + '_bp' for crawler in list_crawler]
    if os.path.exists(path_df_news_articles_score):
        df_news_articles_score = pd.read_csv(path_df_news_articles_score)
    else:
        df_news_articles = pd.read_csv(path_df_news_articles, parse_dates=['pub_date'])
        df_news_articles_score = df_news_articles.copy(deep=True)
        df_news_articles_score[list_crawler_accd] = np.nan
        df_news_articles_score[list_crawler_hwuser] = np.nan
        df_news_articles_score[list_crawler_recent] = np.nan
        df_news_articles_score[list_crawler_bp] = np.nan
    
    df_news_articles_score[list_crawler] = df_news_articles_score[list_crawler].apply(lambda col: col.str.strip())
    mask_not_empty = df_news_articles_score[list_crawler] != ''
    df_news_articles_score[list_crawler] = df_news_articles_score[list_crawler].where(mask_not_empty)
    df_news_articles_score = df_news_articles_score[df_news_articles_score[list_crawler].any(axis=1)]

    ############### remove articles not related to train accident
    dict_answer_choice = {'YES': 1, 'NO': 0}

    list_skip = []
    question = f"Is this article reporting a train accident? Answer only {'/'.join(dict_answer_choice)}"
    df_news_articles_score = check_article(df_news_articles_score, list_skip, list_crawler, list_crawler_accd, dict_answer_choice, question, num_sim=1)

    df_news_articles_score.to_csv(path_df_news_articles_score, index=False)

    ############### remove articles not related to train collision with vehicle/pedestrian
    mask_accd = df_news_articles_score[list_crawler_accd] == 1
    list_skip_accd = df_news_articles_score[~mask_accd.any(axis=1)].index.tolist()
    question = f"Did the train collide with any vehicle/pedestrian? Answer only {'/'.join(dict_answer_choice)}"
    df_news_articles_score = check_article(df_news_articles_score, list_skip_accd, list_crawler, list_crawler_hwuser, dict_answer_choice, question, num_sim=1)

    df_news_articles_score.to_csv(path_df_news_articles_score, index=False)

    ############### remove articles not related to recent train accident
    mask_hwuser = df_news_articles_score[list_crawler_hwuser] == 1
    list_skip_hwuser = df_news_articles_score[~mask_hwuser.any(axis=1)].index.tolist()
    question = f"Did the train accident occur this/last week? Answer only {'/'.join(dict_answer_choice)}"
    df_news_articles_score = check_article(df_news_articles_score, list_skip_hwuser, list_crawler, list_crawler_recent, dict_answer_choice, question, num_sim=1)

    df_news_articles_score.to_csv(path_df_news_articles_score, index=False)
    
    ############### remove articles with boilerplate text
    mask_recent = df_news_articles_score[list_crawler_recent] == 1
    list_skip_recent = df_news_articles_score[~mask_recent.any(axis=1)].index.tolist()
    question = f"Is there any boilerplate text, such as privacy policy, cookie consent, advertising preferences, legal-rights notices, advertisements, UI elements, or unrelated metadata? Answer only {'/'.join(dict_answer_choice)}"
    df_news_articles_score = check_article(df_news_articles_score, list_skip_recent, list_crawler, list_crawler_bp, dict_answer_choice, question, num_sim=1)

    df_news_articles_score.to_csv(path_df_news_articles_score, index=False)

    mask_bp = df_news_articles_score[list_crawler_bp] == 0

    ############### stats
    df_news_articles_score[list_crawler_accd].apply(pd.value_counts)
    df_news_articles_score[list_crawler_hwuser].apply(pd.value_counts)
    df_news_articles_score[list_crawler_recent].apply(pd.value_counts)
    df_news_articles_score[list_crawler_bp].apply(pd.value_counts)

    df_news_articles_score[(mask_recent).any(axis=1)][list_crawler].sample(1).sample(1, axis=1).values

    # df_bp = pd.DataFrame(index=list_crawler, columns=list_crawler)
    # for crawler1 in list_crawler:
    #     for crawler2 in list_crawler:
    #         num_good1 = (df_news_articles_score[crawler1 + '_recent'] == 1)
    #         num_good2 = (df_news_articles_score[crawler2 + '_recent'] == 1)
    #         num_good_both = num_good1 | num_good2
    #         df_bp.loc[crawler1, crawler2] = num_good_both.sum()
    # df_bp = df_bp.where(np.triu(np.ones(df_bp.shape), k=1) == 1, 0)
    # df_bp.drop('tf_html').drop('tf_html', axis=1)
    
    ############### filter news articles
    df_news_articles_filter = df_news_articles_score.copy(deep=True)
    df_news_articles_filter[list_crawler] = df_news_articles_filter[list_crawler].where(mask_accd.values & mask_hwuser.values & mask_recent.values & mask_bp.values)
    df_news_articles_filter = df_news_articles_filter[df_news_articles_filter[list_crawler].any(axis=1)]

    # list_crawler_merge = [crawler.split('_')[0] for crawler in list_crawler[:4]]
    # df_news_articles_filter[list_crawler_merge] = np.nan
    df_news_articles_filter['content'] = np.nan

    ############### select the longest article among different crawlers
    for idx, row in df_news_articles_filter.iterrows():
        content_max_len = max(row[list_crawler].astype(str).values, key=len)
        df_news_articles_filter.loc[idx, 'content'] = content_max_len
    df_news_articles_filter['content'].sample(1).values

    df_news_articles_filter = df_news_articles_filter.drop(list_crawler + list_crawler_accd + list_crawler_hwuser + list_crawler_recent + list_crawler_bp, axis=1)
    df_news_articles_filter.to_csv(path_df_news_articles_filter, index=False)
    return df_news_articles_filter

if __name__ == '__main__':
    
    list_crawler = LIST_CRAWLER
    df_news_record = pd.read_csv(path_df_news_articles, parse_dates=['pub_date'])

    pattern = 'These cookies are necessary for our services to function and cannot be switched off in our systems.'
    df_news_record[list_crawler][df_news_record[list_crawler].apply(lambda col: col.str.contains(pattern, regex=True))].stack().values # check if articles with a specific pattern exist
    df_news_record[list_crawler].loc[1607].values

    # df_news_record_filter[df_news_record_filter[list_crawler].isna().any(axis=1)][list_crawler].sample(1).values # sample a random article to see if there is anything non-article

    # df_news_record_filter[list_crawler].isna().sum()

    # list_crawler_wrong_order = ['np_url', 'np_html', 'tf_url', 'tf_html', 'rd_url', 'rd_html', 'gs_url', 'gs_html']
    # df_news_articles = pd.read_csv(path_df_news_articles, parse_dates=['pub_date'])
    # df_news_articles.columns = ['news_id', 'url', 'pub_date', 'title'] + list_crawler_wrong_order
    # df_news_articles[list_crawler_wrong_order] = df_news_articles[list_crawler]
    # df_news_articles.columns = ['news_id', 'url', 'pub_date', 'title'] + list_crawler
    # df_news_articles[list_crawler]
    # df_news_articles
    # df_news_articles.to_csv(path_df_news_articles, index=False)

    # list_crawler_wrong_order = ['np_url', 'np_html', 'tf_url', 'tf_html', 'rd_url', 'rd_html', 'gs_url', 'gs_html']
    # list_crawler_score_wrong_order = ['score_' + c for c in list_crawler_wrong_order]
    # df_news_articles_score = pd.read_csv(path_df_news_articles_score, parse_dates=['pub_date'])
    # df_news_articles_score.columns = ['news_id', 'url', 'pub_date', 'title'] + list_crawler_wrong_order + list_crawler_score_wrong_order
    # df_news_articles_score[list_crawler_wrong_order] = df_news_articles_score[list_crawler]
    # df_news_articles_score[list_crawler_score_wrong_order] = df_news_articles_score[list_crawler_score]
    # df_news_articles_score.columns = ['news_id', 'url', 'pub_date', 'title'] + list_crawler + list_crawler_score
    # df_news_articles_score[list_crawler]
    # df_news_articles_score[list_crawler_score].sample(10)
    # df_news_articles_score
    # df_news_articles_score.to_csv(path_df_news_articles_score, index=False)