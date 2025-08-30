from tqdm import tqdm
import utils_scrape
import json
import time
import re
import pandas as pd

def scrape_news(path_df_record_news, path_df_news_articles, path_df_record, list_crawlers):
    config_df_record_news = {
        'path': path_df_record_news,
        'columns': ['query1', 'query2', 'county', 'state', 'city', 'highway', 'report_key', 'news_id'],
    }
    config_df_news_articles = {
        'path': path_df_news_articles,
        'columns': ['news_id', 'url', 'pub_date', 'title'] + list_crawlers,
    }

    df_data = pd.read_csv(path_df_record)
    # df_data = df_data[(df_data['State Name'] == 'CALIFORNIA') & (df_data['County Name'] == 'RIVERSIDE')]
    df_data = df_data[df_data['State Name'] == 'CALIFORNIA']
    df_data['hash_id'] = df_data.apply(utils_scrape.hash_row, axis=1)
    df_data['Date'] = pd.to_datetime(df_data['Date'])
    df_data = df_data[df_data['Date'] >= '2000-01-01']
    assert df_data['Report Key'].is_unique
    list_prior_info = ['Report Key', 'Railroad Name', 'Date', 'Nearest Station', 'County Name', 'State Name', 'City Name', 'Highway Name', 'Public/Private', 'Highway User', 'Equipment Type'] # keywords useful for searching
    df_data = df_data.sort_values(['County Name', 'Date'], ascending=[True, False])

    scrape = utils_scrape.Scrape(config_df_record_news, config_df_news_articles)
    if scrape.df_record_news is None or scrape.df_news_articles is None:
        scrape.load_df_record_news()
        scrape.load_df_news_articles()

    # list_query1 = ["train", "amtrak", "locomotive"]
    # list_query2 = ["accident", "incident", "crash", "collide", "hit", "strike", "injure", "kill", "derail"]
    list_query1 = ["train"]
    list_query2 = ["accident"]

    pbar_row = tqdm(df_data.iterrows(), total=df_data.shape[0])
    for i, row in pbar_row:
        row = row.fillna('')
        hash_id, rail_company, date, station, county, state, city, highway, private, vehicle_type, train_type = row[list_prior_info]
        pbar_row.set_description(f'{county}, {city}')

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
                    'report_key': hash_id,
                }
                scrape.set_params(params)

                if scrape.already_scraped():
                    time.sleep(0.001)
                    continue

                feed = scrape.get_RSS()
                assert feed['bozo'] == False
                
                scrape.load_driver()
                df_temp = scrape.get_article(feed)
                scrape.append_df_record_news(df_temp)
                scrape.save_df_record_news()
                scrape.quit_driver()

                if df_temp.shape[0] <= 1:
                    time.sleep(7)
    return scrape.df_record_news, scrape.df_news_articles