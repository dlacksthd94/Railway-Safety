from tqdm import tqdm
import utils_scrape
import json
import time
import re
import pandas as pd

DATA_FOLDER = 'data/'
FN_DF = 'df_news.csv'
FN_DF_DATA = '250424 Highway-Rail Grade Crossing Incident Data (Form 57).csv'
COLUMNS = ['query1', 'query2', 'county', 'state', 'city', 'highway', 'incident_id', 'news_id', 'url', 'pub_date', 'title', 'np_url', 'tf_url', 'rd_url', 'gs_url', 'np_html', 'tf_html', 'rd_html', 'gs_html']

path_df=DATA_FOLDER + FN_DF
path_df_data = DATA_FOLDER + FN_DF_DATA

df_data = pd.read_csv(path_df_data)
# df_data = df_data[(df_data['State Name'] == 'CALIFORNIA') & (df_data['County Name'] == 'RIVERSIDE')]
df_data = df_data[df_data['State Name'] == 'CALIFORNIA']
df_data['hash_id'] = df_data.apply(utils_scrape.hash_row, axis=1)
df_data['Date'] = pd.to_datetime(df_data['Date'])
df_data = df_data[df_data['Date'] >= '2000-01-01']
list_prior_info = ['hash_id', 'Railroad Name', 'Date', 'Nearest Station', 'County Name', 'State Name', 'City Name', 'Highway Name', 'Public/Private', 'Highway User', 'Equipment Type'] # keywords useful for searching
df_data = df_data.sort_values(['County Name', 'Date'], ascending=[True, False])

scrape = utils_scrape.Scrape(COLUMNS)
if scrape.df is None:
    scrape.load_df(path_df)

# list_query1 = ["train", "amtrak", "locomotive"]
# list_query2 = ["accident", "incident", "crash", "collide", "hit", "strike", "injure", "kill", "derail"]
list_query1 = ["train"]
list_query2 = ["accident"]

pbar_row = tqdm(df_data.iterrows(), total=df_data.shape[0])
for i, row in pbar_row:
    row = row.fillna('')
    hash_id, rail_company, date, station, county, state, city, highway, private, vehicle_type, train_type = row[list_prior_info]
    pbar_row.set_description(f'{county}, {city}, {highway}')

    pbar_query1 = tqdm(list_query1, leave=False)
    for query1 in pbar_query1:
        pbar_query1.set_description(query1)
        pbar_query2 = tqdm(list_query2, leave=False)
        for query2 in pbar_query2:
            pbar_query2.set_description(query2)
            
            params = {
                'query1': query1,
                'query2': query2,
                # 'rail_company': rail_company,
                # 'station': station,
                'county': county,
                'state': state,
                'city': city,
                'highway': highway,
                'private': private,
                # 'vehicle_type': vehicle_type,
                # 'train_type': train_type,
                'date_from': (date - pd.Timedelta(days=7)).strftime('%Y-%m-%d'),
                'date_to': (date + pd.Timedelta(days=7)).strftime('%Y-%m-%d'),
                'incident_id': hash_id,
            }
            scrape.set_params(params)

            if scrape.already_scraped():
                time.sleep(0.01)
                continue

            feed = scrape.get_RSS()
            assert feed['bozo'] == False
            
            scrape.load_driver()
            df_temp = scrape.get_article(feed)
            scrape.append_df(df_temp)
            scrape.save_df(path_df)
            scrape.quit_driver()

            time.sleep(10)