import pandas as pd
import sys
import os
import spacy
from tqdm import tqdm
import datetime
import re
import simcse
from sklearn.metrics.pairwise import cosine_similarity

DATA_FOLDER = 'data/'
pd.set_option('display.max_columns', 10, 'display.max_rows', 150)

fn_df = DATA_FOLDER + 'Highway-Rail_Grade_Crossing_Accident_Data__Form_57__20240925.csv'
if os.path.exists(fn_df):
    df = pd.read_csv(fn_df)
    df['Date'] = pd.to_datetime(df['Date'])
    # df[df['Date'] > '2010-01-01']
####################

fn_df_news = DATA_FOLDER + 'df_news_state.csv'
df_news = pd.read_csv(fn_df_news)
df_news['date'] = pd.to_datetime(df_news['date'])

days_of_week = {
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6
}

def days_between_day_names(day1, day2):
    days_difference = (days_of_week[day1] - days_of_week[day2]) % 7
    return days_difference

nlp = spacy.load("en_core_web_trf")

# ################### FILTERING ARTILES NOT RELATED TO ACTUAL CRASH #########################
# model = simcse.SimCSE('princeton-nlp/sup-simcse-roberta-large')
# list_embedding = []
# for i, row in tqdm(df_news.iterrows(), total=df_news.shape[0]):
#     content = row['content'].replace('\n\n', ' ')
#     doc = nlp(content)
#     embedding = model.encode(list(map(lambda s: s.text, doc.sents)), return_numpy=True)
#     list_embedding.append(embedding)
#     break
# df_news['embedding'] = list_embedding

# sentences = ['A train crashed into a car.', 'A train crashed into a truck.', 'A train crashed into a trailer.', 'A train crashed into a man.', 'A train crashed into a woman.', 'A train crashed into a motorcycle.', 'A train crashed into a bicycle.', 'A train crashed into a bus.', 'A train crashed into a van.', 'A train collided with a car.', 'A train collided with a truck.', 'A train collided with a trailer.', 'A train collided with a man.', 'A train collided with a woman.', 'A train collided with a motorcycle.', 'A train collided with a bicycle.', 'A train collided with a bus.', 'A train collided with a van.', 'A train hit a car.', 'A train hit a truck.', 'A train hit a trailer.', 'A train hit a man.', 'A train hit a woman.', 'A train hit a motorcycle.', 'A train hit a bicycle.', 'A train hit a bus.', 'A train hit a van.', 'A train smashed into a car.', 'A train smashed into a truck.', 'A train smashed into a trailer.', 'A train smashed into a man.', 'A train smashed into a woman.', 'A train smashed into a motorcycle.', 'A train smashed into a bicycle.', 'A train smashed into a bus.', 'A train smashed into a van.', 'A train struck a car.', 'A train struck a truck.', 'A train struck a trailer.', 'A train struck a man.', 'A train struck a woman.', 'A train struck a motorcycle.', 'A train struck a bicycle.', 'A train struck a bus.', 'A train struck a van.', 'A train ran into a car.', 'A train ran into a truck.', 'A train ran into a trailer.', 'A train ran into a man.', 'A train ran into a woman.', 'A train ran into a motorcycle.', 'A train ran into a bicycle.', 'A train ran into a bus.', 'A train ran into a van.']
# target_sentence = model.encode(sentences, return_numpy=True)
# cosine_similarity(target_sentence, target_sentence)
# max_score = df_news['embedding'].apply(lambda l: cosine_similarity(l, target_sentence).mean(axis=1).max())
# df_news['score'] = max_score
# df_news[(df_news['score'] <= 0.5) & (df_news['score'] <= 0.5)]['content'].iloc[11]
# df_news.iloc[36]['title']
# df_news.iloc[36]['img_desc']

# list(nlp(df_news['content'][0].replace('\n\n', ' ')).sents)[7]

""" """ """ """ """ """ """ ARRANGE NAMED ENTITIES """ """ """ """ """ """ """
for _, row in tqdm(df_news.iterrows(), total=df_news.shape[0]):
    content = row['content']
    pub_date = row['pub_date']
    doc = nlp(content)
    df_ner = pd.DataFrame([{'word': ent.text, 'label': ent.label_} for ent in doc.ents])
    if not df_ner.empty:
        df_ner = df_ner[~df_ner['word'].str.lower().duplicated()]
        # extract date
        sr_dt_ne = df_ner[df_ner['label'].isin(['DATE', 'TIME'])]['word']
        sr_acdt_date = pd.to_datetime(sr_dt_ne, errors='coerce').dropna()
        if not sr_acdt_date.empty:
            acdt_date = sr_acdt_date.values[0]
        else:
            pat = '|'.join(days_of_week)
            text = ' '.join(df_ner['word'].values)
            search_result = re.search(pat, text)
            if search_result:
                acdt_date_day_name = search_result.group()
                pub_date_day_name = pub_date.day_name()
                days_delta = days_between_day_names(pub_date_day_name, acdt_date_day_name)
                acdt_date = pub_date - pd.Timedelta(days=days_delta)
            else:
                continue
        df_match = df[df['Date'] == acdt_date]
        if df_match.empty:
            pass
        else:
            if df_match.index.size <= 1:
                df.loc[df['Date'] == acdt_date, 'Content'] = content
            else:
                sr_fg_ne = df_ner[df_ner['label'].isin(['FAC', 'GPE'])]['word'].str.lower()
                if not sr_fg_ne.empty:
                    num_loc_mtch = df_match[['County Name', 'City Name', 'Highway Name']].fillna('nan')\
                        .apply(lambda sr: sr.str.lower().str.match('|'.join(sr_fg_ne))).sum(axis=1)
                    if num_loc_mtch.max() > 0 and (num_loc_mtch == num_loc_mtch.max()).sum() == 1:
                        idx_matched = num_loc_mtch.sort_values(ascending=False).index[0]
                        df.loc[idx_matched, 'Content'] = content
df[~df['Content'].isna()]['Narrative'] # records merged with articles already have narratives
df[~df['Content'].isna()][['Narrative', 'Content']].iloc[0].values
df[~df['Content'].isna()].iloc[0].to_dict()

fn_df = DATA_FOLDER + 'df.csv'
df.to_csv(fn_df, index=False)