from tqdm import tqdm
import utils_scrape
import json
import time

DATA_FOLDER = 'data/'
FN_DF = 'df_news.csv'
COLUMNS = ['query1', 'query2', 'state', 'county', 'city', 'id', 'url', 'pub_date', 'title', 'content']

path_json_county_city = DATA_FOLDER + 'dict_California_county_city.json'
with open(path_json_county_city, 'r') as file:
    dict_California_county_city = json.load(file)
    dict_California_county_city = {k.strip(' County'): v for k, v in dict_California_county_city.items() if 'County' in k}
    for k ,v in dict_California_county_city.items():
        v.insert(0, '-')

path_json_list_keywords = DATA_FOLDER + 'dict_list_keywords.json'
with open(path_json_list_keywords, "r") as file:
    dict_list_keywords = json.load(file)

scrape = utils_scrape.Scrape(COLUMNS)
if scrape.df is None:
    path_df_cal=DATA_FOLDER + FN_DF
    scrape.load_df(path_df_cal)

list_query1 = dict_list_keywords["list_query1"]
list_query2 = dict_list_keywords["list_query2"]
list_state = dict_list_keywords["list_state"]
list_county = dict_California_county_city.keys()

for query1 in tqdm(list_query1, leave=False):
    for query2 in tqdm(list_query2, leave=False):
        for state in tqdm(list_state, leave=False):
            for county in tqdm(list_county, leave=False):
                list_city = dict_California_county_city[county]
                for city in tqdm(list_city, leave=False):
                    params = {
                        'query1': query1,
                        'query2': query2,
                        'state': state,
                        'county': county,
                        'city': city,
                        'date_from': "2000-01-01",
                        'date_to': "2024-12-31",
                    }
                    scrape.set_params(params)

                    if scrape.already_scraped():
                        continue

                    feed = scrape.get_RSS()
                    f"{query1}, {query2}, {state}, {county}, {city}, {len(feed['entries'])}"
                    assert feed['bozo'] == False
                    
                    scrape.load_driver()
                    df_temp = scrape.get_article(feed)
                    scrape.append_df(df_temp)
                    scrape.save_df(path_df_cal)
                    scrape.quit_driver()

                    time.sleep(5)
