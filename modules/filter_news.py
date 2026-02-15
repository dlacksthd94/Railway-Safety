from transformers import pipeline
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import os
import torch
import gc
from itertools import combinations
from google import genai
from .utils import text_binary_classification, text_generation, select_generate_func, desanitize_model_path

pd.set_option('display.max_colwidth', 30)

# MODEL_PATH = 'microsoft/Phi-4-mini-instruct'
# MODEL_PATH = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
MODEL_PATH = 'Qwen/Qwen2.5-7B-Instruct-1M'
# DEVICE = 'cuda:1'
DEVICE = 'auto'


def check_article(df_news_articles_score, list_skip, col_crawlers, col_crawlers_score, dict_answer_choice, question, num_sim=1):
    """
    classify each scraped article (per crawler) as reporting a train accident and store an averaged binary score.
    """
    assert len(set(dict_answer_choice.values())) == len(dict_answer_choice.values())
    pipe = pipeline(model=MODEL_PATH, device_map=DEVICE) # type: ignore

    for idx, row in tqdm(df_news_articles_score.iterrows(), total=df_news_articles_score.shape[0]):
        if idx in list_skip:
            continue
        if row[col_crawlers_score].notna().any():
            continue
        sr_content = row[col_crawlers]
        dict_column_score = {crawler_score: float('nan') for crawler_score in col_crawlers_score}
        for col_crawler, col_crawler_score in tqdm(zip(col_crawlers, col_crawlers_score), total=len(col_crawlers), leave=False):
            content = sr_content[col_crawler]
            if pd.isna(content):
                dict_column_score[col_crawler_score] = dict_answer_choice['NO']
            else:
                try:
                    prompt = f"Context:\n{content}\n\nQuestion:\n{question}\n\nAnswer:\n"
                    list_answer_map = text_binary_classification(pipe, prompt, dict_answer_choice, num_sim=num_sim)
                    if len(list_answer_map) > 0:
                        dict_column_score[col_crawler_score] = np.mean(list_answer_map, dtype=float)
                except:
                    pass
        df_news_articles_score.loc[idx, col_crawlers_score] = dict_column_score.values()

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return df_news_articles_score


def check_article_realtime(cfg, df_news_articles_score, mask_skip, col_crawlers, col_crawlers_score, dict_answer_choice, question, num_sim=1):
    """
    classify each scraped article (per crawler) as reporting a train accident and store an averaged binary score.
    """
    assert len(set(dict_answer_choice.values())) == len(dict_answer_choice.values())

    api, model_path = 'Google', 'gemini-2.5-flash-lite'
    model_path = desanitize_model_path(model_path)
    client = genai.Client(api_key=cfg.apikey.google)

    generate_func = select_generate_func(api)

    for (idx, row), mask_row in tqdm(zip(df_news_articles_score.iterrows(), mask_skip), total=df_news_articles_score.shape[0]):
        if mask_row.all():
            continue
        sr_content = row[col_crawlers]
        dict_column_score = {crawler_score: float('nan') for crawler_score in col_crawlers_score}
        for col_crawler, col_crawler_score, mask in tqdm(zip(col_crawlers, col_crawlers_score, mask_row), total=len(col_crawlers), leave=False):
            if mask:
                continue
            content = sr_content[col_crawler]
            if pd.isna(content):
                dict_column_score[col_crawler_score] = dict_answer_choice['NO']
            else:
                prompt = f"Context:\n{content}\n\nQuestion:\n{question}\n\nAnswer:\n"
                content = [prompt]
                
                answers_map = []
                for _ in range(num_sim):
                    output = generate_func(client, model_path, content, generation_config=None)

                    answer = output.upper() # type: ignore
                    if answer in dict_answer_choice:
                        answer_map = dict_answer_choice[answer]
                    else:
                        answer_map = float('nan')
                    answers_map.append(answer_map)

                dict_column_score[col_crawler_score] = np.mean(answers_map) if not pd.isna(answers_map).all() else float('nan') # type: ignore
        df_news_articles_score.loc[idx, col_crawlers_score] = dict_column_score.values()

    return df_news_articles_score


def get_accident_date(cfg, content, question, dict_answer_choice):
    assert len(set(dict_answer_choice.values())) == len(dict_answer_choice.values())

    api, model_path = 'Google', 'gemini-2.5-flash-lite'
    model_path = desanitize_model_path(model_path)
    client = genai.Client(api_key=cfg.apikey.google)

    generate_func = select_generate_func(api)

    prompt = f"Context:\n{content}\n\nQuestion:\n{question}\n\nAnswer:\n"
    content = [prompt]

    output = generate_func(client, model_path, content, generation_config=None)

    return output

def filter_news(cfg) -> pd.DataFrame:
    col_crawlers = list(cfg.scrp.news_crawlers)
    col_crawlers_accd = [col + '_accd' for col in col_crawlers]
    col_crawlers_hwuser = [crawler + '_hwuser' for crawler in col_crawlers]
    col_crawlers_recent = [crawler + '_recent' for crawler in col_crawlers]
    col_crawlers_bp = [crawler + '_bp' for crawler in col_crawlers]
    if os.path.exists(cfg.path.df_news_articles_score):
        df_news_articles_score = pd.read_csv(cfg.path.df_news_articles_score)
    else:
        df_news_articles = pd.read_csv(cfg.path.df_news_articles, parse_dates=['pub_date'])
        df_news_articles_score = df_news_articles.copy(deep=True)
        df_news_articles_score[col_crawlers_accd] = np.nan
        df_news_articles_score[col_crawlers_hwuser] = np.nan
        df_news_articles_score[col_crawlers_recent] = np.nan
        df_news_articles_score[col_crawlers_bp] = np.nan
    
    df_news_articles_score[col_crawlers] = df_news_articles_score[col_crawlers].apply(lambda col: col.str.strip())
    mask_not_empty = df_news_articles_score[col_crawlers] != ''
    df_news_articles_score[col_crawlers] = df_news_articles_score[col_crawlers].where(mask_not_empty)
    df_news_articles_score = df_news_articles_score[df_news_articles_score[col_crawlers].any(axis=1)]

    ############### remove articles not related to train accident
    dict_answer_choice = {'YES': 1, 'NO': 0}

    list_skip = []
    question = f"Is this article reporting a train accident? Answer only {'/'.join(dict_answer_choice)}"
    df_news_articles_score = check_article(df_news_articles_score, list_skip, col_crawlers, col_crawlers_accd, dict_answer_choice, question, num_sim=1)

    df_news_articles_score.to_csv(cfg.path.df_news_articles_score, index=False)

    ############### remove articles not related to train collision with vehicle/pedestrian
    mask_accd = df_news_articles_score[col_crawlers_accd] == 1
    list_skip_accd = df_news_articles_score[~mask_accd.any(axis=1)].index.tolist()
    question = f"Did the train collide with any vehicle/pedestrian? Answer only {'/'.join(dict_answer_choice)}"
    df_news_articles_score = check_article(df_news_articles_score, list_skip_accd, col_crawlers, col_crawlers_hwuser, dict_answer_choice, question, num_sim=1)

    df_news_articles_score.to_csv(cfg.path.df_news_articles_score, index=False)

    ############### remove articles not related to recent train accident
    mask_hwuser = df_news_articles_score[col_crawlers_hwuser] == 1
    list_skip_hwuser = df_news_articles_score[~mask_hwuser.any(axis=1)].index.tolist()
    question = f"Did the train accident occur this/last week? Answer only {'/'.join(dict_answer_choice)}"
    df_news_articles_score = check_article(df_news_articles_score, list_skip_hwuser, col_crawlers, col_crawlers_recent, dict_answer_choice, question, num_sim=1)

    df_news_articles_score.to_csv(cfg.path.df_news_articles_score, index=False)
    
    ############### remove articles with boilerplate text
    mask_recent = df_news_articles_score[col_crawlers_recent] == 1
    list_skip_recent = df_news_articles_score[~mask_recent.any(axis=1)].index.tolist()
    question = f"Is there any boilerplate text, such as privacy policy, cookie consent, advertising preferences, legal-rights notices, advertisements, UI elements, or unrelated metadata? Answer only {'/'.join(dict_answer_choice)}"
    df_news_articles_score = check_article(df_news_articles_score, list_skip_recent, col_crawlers, col_crawlers_bp, dict_answer_choice, question, num_sim=1)

    df_news_articles_score.to_csv(cfg.path.df_news_articles_score, index=False)

    mask_bp = df_news_articles_score[col_crawlers_bp] == 0

    ############### stats
    df_news_articles_score[col_crawlers_accd].apply(pd.value_counts)
    df_news_articles_score[col_crawlers_hwuser].apply(pd.value_counts)
    df_news_articles_score[col_crawlers_recent].apply(pd.value_counts)
    df_news_articles_score[col_crawlers_bp].apply(pd.value_counts)

    df_news_articles_score[(mask_recent).any(axis=1)][col_crawlers].sample(1).sample(1, axis=1).values

    # df_bp = pd.DataFrame(index=col_crawlers, columns=col_crawlers)
    # for crawler1 in col_crawlers:
    #     for crawler2 in col_crawlers:
    #         num_good1 = (df_news_articles_score[crawler1 + '_recent'] == 1)
    #         num_good2 = (df_news_articles_score[crawler2 + '_recent'] == 1)
    #         num_good_both = num_good1 | num_good2
    #         df_bp.loc[crawler1, crawler2] = num_good_both.sum()
    # df_bp = df_bp.where(np.triu(np.ones(df_bp.shape), k=1) == 1, 0)
    # df_bp.drop('tf_html').drop('tf_html', axis=1)
    
    ############### filter news articles
    df_news_articles_filter = df_news_articles_score.copy(deep=True)
    df_news_articles_filter[col_crawlers] = df_news_articles_filter[col_crawlers].where(mask_accd.values & mask_hwuser.values & mask_recent.values & mask_bp.values)
    df_news_articles_filter = df_news_articles_filter[df_news_articles_filter[col_crawlers].any(axis=1)]

    # col_crawlers_merge = [crawler.split('_')[0] for crawler in col_crawlers[:4]]
    # df_news_articles_filter[col_crawlers_merge] = np.nan
    df_news_articles_filter['content'] = np.nan

    ############### select the longest article among different crawlers
    for idx, row in df_news_articles_filter.iterrows():
        content_max_len = max(row[col_crawlers].astype(str).values, key=len)
        df_news_articles_filter.loc[idx, 'content'] = content_max_len
    df_news_articles_filter['content'].sample(1).values

    df_news_articles_filter = df_news_articles_filter.drop(col_crawlers + col_crawlers_accd + col_crawlers_hwuser + col_crawlers_recent + col_crawlers_bp, axis=1)
    df_news_articles_filter.to_csv(cfg.path.df_news_articles_filter, index=False)
    return df_news_articles_filter


def filter_news_realtime(cfg, start_date, state, end_date) -> pd.DataFrame:
    col_crawlers = list(cfg.scrp.news_crawlers)
    # col_crawlers_clean = [crawler + '_clean' for crawler in col_crawlers]
    col_crawlers_train = [crawler + '_train' for crawler in col_crawlers]
    col_crawlers_state = [crawler + '_state' for crawler in col_crawlers]
    if os.path.exists(cfg.path.df_news_articles_realtime_score):
        df_news_articles_score = pd.read_csv(cfg.path.df_news_articles_realtime_score, parse_dates=['pub_date'])
        df_news_articles = pd.read_csv(cfg.path.df_news_articles_realtime, parse_dates=['pub_date'])
        df_news_articles_new = df_news_articles[~df_news_articles['news_id'].isin(df_news_articles_score['news_id'])].copy(deep=True)
        # df_news_articles_new[col_crawlers_clean] = np.nan
        df_news_articles_new[col_crawlers_train] = np.nan
        df_news_articles_new[col_crawlers_state] = np.nan
        df_news_articles_score = pd.concat([df_news_articles_score, df_news_articles_new], ignore_index=True)
    else:
        df_news_articles = pd.read_csv(cfg.path.df_news_articles_realtime, parse_dates=['pub_date'])
        df_news_articles_score = df_news_articles.copy(deep=True)
        # df_news_articles_score[col_crawlers_clean] = np.nan
        df_news_articles_score[col_crawlers_train] = np.nan
        df_news_articles_score[col_crawlers_state] = np.nan

    ### remove empty articles
    df_news_articles_score[col_crawlers] = df_news_articles_score[col_crawlers].apply(lambda col: col.str.strip())
    mask_not_empty = df_news_articles_score[col_crawlers] != ''
    df_news_articles_score[col_crawlers] = df_news_articles_score[col_crawlers].where(mask_not_empty)
    df_news_articles_score = df_news_articles_score[df_news_articles_score[col_crawlers].any(axis=1)]


    ### remove articles not related to train accident
    dict_answer_choice = {'YES': 1, 'NO': 0}

    ### remove articles out of recent date range
    # mask_skip_date = (df_news_articles_score['pub_date'] < start_date) | (df_news_articles_score['pub_date'] > end_date)
    # mask_skip_date = np.broadcast_to(mask_skip_date.values[:, np.newaxis], (mask_skip_date.shape[0], len(col_crawlers)))
    # df_news_articles_score = df_news_articles_score.loc[[2, 93, 101, 116, 138]]
    mask_skip_date = np.zeros((df_news_articles_score.shape[0], len(col_crawlers)), dtype=bool)

    # ### remove articles with boilerplate text
    # mask_clean_done = df_news_articles_score[col_crawlers_clean].notna()
    # mask_clean_done = np.broadcast_to(mask_clean_done.any(axis=1).values[:, np.newaxis], (mask_clean_done.shape[0], len(col_crawlers))) # type: ignore
    # mask_skip = mask_skip_date | mask_clean_done
    # question = f"Is this a mainly clean news article without any unrelated text such as privacy policy, cookie consent, advertising preferences, advertisements, UI elements, etc.? Answer only {'/'.join(dict_answer_choice)}"
    # df_news_articles_score = check_article_realtime(cfg, df_news_articles_score, mask_skip, col_crawlers, col_crawlers_clean, dict_answer_choice, question, num_sim=1)
    # df_news_articles_score[col_crawlers].where(df_news_articles_score[col_crawlers_clean].values == 1)

    # df_news_articles_score.to_csv(cfg.path.df_news_articles_realtime_score, index=False)

    ### remove articles not related to train accident
    mask_train_done = df_news_articles_score[col_crawlers_train].notna()
    mask_train_done = np.broadcast_to(mask_train_done.any(axis=1).values[:, np.newaxis], (mask_train_done.shape[0], len(col_crawlers))) # type: ignore
    # mask_skip_not_clean = (df_news_articles_score[col_crawlers_clean] != 1).values
    # mask_skip = mask_skip_date | mask_train_done | mask_skip_not_clean
    mask_skip = mask_skip_date | mask_train_done
    question = f"Does this article report a train accident? Answer only {'/'.join(dict_answer_choice)}"
    df_news_articles_score = check_article_realtime(cfg, df_news_articles_score, mask_skip, col_crawlers, col_crawlers_train, dict_answer_choice, question, num_sim=3)
    df_news_articles_score[col_crawlers].where(df_news_articles_score[col_crawlers_train].values == 1)

    df_news_articles_score.to_csv(cfg.path.df_news_articles_realtime_score, index=False)

    ### remove articles not related to US STATE
    mask_state_done = df_news_articles_score[col_crawlers_state].notna()
    mask_state_done = np.broadcast_to(mask_state_done.any(axis=1).values[:, np.newaxis], (mask_state_done.shape[0], len(col_crawlers))) # type: ignore
    mask_skip_is_train = (df_news_articles_score[col_crawlers_train] > 0).values
    mask_skip = mask_skip_date | mask_state_done | ~mask_skip_is_train
    question = f"Did the train accident occur in US {state}? Answer only {'/'.join(dict_answer_choice)}"
    df_news_articles_score = check_article_realtime(cfg, df_news_articles_score, mask_skip, col_crawlers, col_crawlers_state, dict_answer_choice, question, num_sim=3)
    df_news_articles_score[col_crawlers].where(df_news_articles_score[col_crawlers_state].values == 1)

    df_news_articles_score.to_csv(cfg.path.df_news_articles_realtime_score, index=False)

    mask_valid_final = (df_news_articles_score[col_crawlers_state] == 1).values

    ### filter news articles
    df_news_articles_filter = df_news_articles_score.copy(deep=True)
    df_news_articles_filter[col_crawlers] = df_news_articles_filter[col_crawlers].where(mask_valid_final)
    df_news_articles_filter = df_news_articles_filter[df_news_articles_filter[col_crawlers].any(axis=1)].copy(deep=True)
    df_news_articles_filter['content'] = np.nan
    df_news_articles_filter['content'] = df_news_articles_filter['content'].astype('object')

    ### select the longest article among different crawlers
    df_news_articles_filter['selected_crawler'] = np.nan
    df_news_articles_filter['selected_crawler'] = df_news_articles_filter['selected_crawler'].astype('object')
    for idx, row in df_news_articles_filter.iterrows():
        selected_crawler = row[col_crawlers].str.len().idxmax()
        content_max_len = row[selected_crawler]
        df_news_articles_filter.loc[idx, 'selected_crawler'] = selected_crawler
        df_news_articles_filter.loc[idx, 'content'] = content_max_len
    df_news_articles_filter['content'].sample(1).values
    df_news_articles_score[col_crawlers_train]
    df_news_articles_score[col_crawlers_state]
    df_news_articles_score.loc[138, 'tf_html'] # 2, 93, 101, 116, 138

    # df_news_articles_filter = df_news_articles_filter.drop(col_crawlers + col_crawlers_clean + col_crawlers_train + col_crawlers_state, axis=1)
    df_news_articles_filter = df_news_articles_filter.drop(col_crawlers + col_crawlers_train + col_crawlers_state, axis=1)

    ### identify the accident date and time
    df_news_articles_filter['accident_date'] = np.nan
    df_news_articles_filter['accident_date'] = df_news_articles_filter['accident_date'].astype('object')
    for idx, row in tqdm(df_news_articles_filter.iterrows()):
        content = row['content']
        pub_date = row['pub_date']
        question = f"The article was published on {pub_date}. When is the date of the train accident? Answer in YYYY-MM-DD format. If the date cannot be identified, answer 'unknown'."
        accident_date = get_accident_date(cfg, content, question, dict_answer_choice)
        df_news_articles_filter.loc[idx, 'accident_date'] = accident_date
        print()
        print(pub_date, pub_date.day_name())
        print(content)
        print(accident_date)
        print()

    df_news_articles_filter.to_csv(cfg.path.df_news_articles_realtime_filter, index=False)

    return df_news_articles_filter


if __name__ == '__main__':
    pass
    # df_news_record = pd.read_csv(cfg.path.df_news_articles, parse_dates=['pub_date'])

    # pattern = 'These cookies are necessary for our services to function and cannot be switched off in our systems.'
    # df_news_record[col_crawlers][df_news_record[col_crawlers].apply(lambda col: col.str.contains(pattern, regex=True))].stack().values # check if articles with a specific pattern exist
    # df_news_record[col_crawlers].loc[1607].values

    # df_news_record_filter[df_news_record_filter[col_crawlers].isna().any(axis=1)][col_crawlers].sample(1).values # sample a random article to see if there is anything non-article

    # df_news_record_filter[col_crawlers].isna().sum()

    # no_article_patterns = [
    #     '^\s+$',
    #     'Press & Hold to confirm you',
    #     'You are not authorized to access',
    #     'Edge: Too Many Requests',
    #     '403 Forbidden',
    #     'These cookies are necessary for our services to function',
    #     'Selling, Sharing, Targeted Advertising',
    #     'Please enable JS and disable any ad blocker',
    #     'We may use personal information to support',
    #     'Access to this site has been denied.',
    #     'This website is using a security service to protect itself from online attacks',
    #     'we provide online advertising services that use cookies and similar technologies to collect information',
    #     "We won't sell or share your personal information to inform the ads you see.",
    #     'we process personal information to inform which ads you see on our services',
    #     'Enable JavaScript and cookies to continue',
    #     "We won't sell your personal information to inform the ads you see.",
    #     'When you visit our website, we store cookies on your browser to collect information.',
    #     'Get the facts every day.',
    #     'Close Get email notifications',

    #     'Click here if you wish to opt out and view the full site.',
    #     'This website uses cookies and other tracking technologies',
    #     'Attackers might be trying to steal your information',
    #     'We use cookies to help you navigate efficiently and perform certain functions',
    #     'We sell and/or share your personal information/data',
    #     'you have the right to opt-out of Targeted Advertising',
    #     'This website utilizes technologies such as cookies to enable essential site functionality',
    #     'we and our ad partners collect certain information from our visitors through cookies and similar technologies',
    #     'State Alabama Alaska Arizona Arkansas California Colorado Connecticut Delaware Florida Georgia Hawaii Idaho Illinois Indiana Iowa Kansas Kentucky Louisiana Maine Maryland Massachusetts Michigan Minnesota Mississippi Missouri Montana Nebraska Nevada New Hampshire New Jersey New Mexico New York North Carolina North Dakota Ohio Oklahoma Oregon Pennsylvania Rhode Island South Carolina South Dakota Tennessee Texas Utah Vermont Virginia Washington Washington D.C. West Virginia Wisconsin Wyoming',
    #     'remaining of\n\nThank you for reading! On your next view you will be asked to log in to your subscriber account',
    #     'Attackers can see and change information you send or receive from the site.\n\ninformation you send or receive from the site.',
    #     "Not at all. It just seems like a lot of back-and-forth talk.\n\nYes. I'm growing very worried over what might happen.",
    #     'Fate of OC Judge who shot and killed his wife once again in the hands of a jury',
    #     'Please email your obituary to obituary@chicoer.com',
    # ]
    # pattern = '|'.join(no_article_patterns)