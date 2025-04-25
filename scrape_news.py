from tqdm import tqdm
from utils_scrape import Scrape, VerificationError
import json
import time
import re
from bs4 import BeautifulSoup

DATA_FOLDER = 'data/'
FN_DF = 'df_news.csv'
COLUMNS = ['query1', 'query2', 'state', 'county', 'city', 'id', 'url', 'pub_date', 'title', 'np_url', 'tf_url', 'rd_url', 'gs_url', 'np_html', 'tf_html', 'rd_html', 'gs_html']

path_df=DATA_FOLDER + FN_DF
path_json_county_city = DATA_FOLDER + 'dict_California_county_city.json'
path_json_list_keywords = DATA_FOLDER + 'dict_list_keywords.json'

with open(path_json_county_city, 'r') as file:
    dict_California_county_city = json.load(file)
    dict_California_county_city = {k.replace(' County', ''): v for k, v in dict_California_county_city.items() if 'County' in k}
    for k ,v in dict_California_county_city.items():
        v.insert(0, '-')

with open(path_json_list_keywords, "r") as file:
    dict_list_keywords = json.load(file)

scrape = Scrape(COLUMNS)
if scrape.df is None:
    scrape.load_df(path_df)

list_query1 = dict_list_keywords["list_query1"]
list_query2 = dict_list_keywords["list_query2"]
list_state = dict_list_keywords["list_state"]
list_county = dict_California_county_city.keys()

pbar_query1 = tqdm(list_query1, leave=False)
for query1 in pbar_query1:
    pbar_query1.set_description(query1)
    pbar_query2 = tqdm(list_query2, leave=False)
    for query2 in pbar_query2:
        pbar_query2.set_description(query2)
        pbar_state = tqdm(list_state, leave=False)
        for state in pbar_state:
            pbar_state.set_description(state)
            pbar_county = tqdm(list_county, leave=False)
            for county in pbar_county:
                pbar_county.set_description(county)
                list_city = dict_California_county_city[county]
                pbar_city = tqdm(list_city, leave=False)
                for city in pbar_city:
                    pbar_city.set_description(city)
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
                        time.sleep(0.001)
                        continue

                    feed = scrape.get_RSS()
                    f"{query1}, {query2}, {state}, {county}, {city}, {len(feed['entries'])}"
                    assert feed['bozo'] == False
                    
                    scrape.load_driver()
                    df_temp = scrape.get_article(feed)
                    scrape.append_df(df_temp)
                    scrape.save_df(path_df)
                    scrape.quit_driver()

                    time.sleep(3)

# scrape = Scrape(COLUMNS)
# scrape.load_df(path_df)
# scrape.load_driver()

# human_verification = [
#     'Press & Hold to confirm you are\n\na human',
#     'This website is using a security service to protect itself from online attacks.',
#     'Verify you are human by completing the action below',
#     "Sorry, we have to make sure you're a human before we can show you this page."
# ]
# pattern = r'|'.join(human_verification)
# indices_rescrape = scrape.df[scrape.df['content'].str.contains(pattern, na=False)].index
# print(f"num of articles to re-scrape: {len(indices_rescrape)}")

# for idx in indices_rescrape:
#     url = scrape.df.loc[idx, 'url']

#     try: # if the page is not loaded in 20 seconds, an error occurs.
#         content, redirect_url = scrape.extract_content(url)
#         if re.search(pattern, content):
#             time.sleep(60)
#             raise VerificationError
#         # scrape.press_and_hold()
#         scrape.df.loc[idx, 'content'] = content
#         scrape.save_df(path_df)
#     except:
#         scrape.quit_driver()
#         scrape.load_driver()

#     time.sleep(10)