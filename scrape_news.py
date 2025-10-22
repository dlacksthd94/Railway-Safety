from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.common.by import By
# from selenium.webdriver.common.action_chains import ActionChains

import feedparser
from urllib.parse import urlencode
import newspaper
import trafilatura
import readability
import requests
from bs4 import BeautifulSoup
import goose3

import pandas as pd
import time
from tqdm import tqdm
import os
import subprocess
import platform
import copy
from config import Config, TableConfig

TIMEOUT = 5
CONFIG_NP = newspaper.Config()
CONFIG_NP.request_timeout = TIMEOUT
CONFIG_TF = copy.deepcopy(trafilatura.settings.DEFAULT_CONFIG) # type: ignore
CONFIG_TF['DEFAULT']['DOWNLOAD_TIMEOUT'] = str(TIMEOUT)

class Scrape:
    driver: webdriver.Chrome
    df_record_news: pd.DataFrame
    df_news_articles: pd.DataFrame

    def __init__(self, config_df_record_news: TableConfig, config_df_news_articles: TableConfig):
        self.config_df_record_news = config_df_record_news
        self.config_df_news_articles = config_df_news_articles
        self.query1 = None
        self.query2 = None
        self.rail_company = None
        self.station = None
        self.county = None
        self.state = None
        self.city = None
        self.highway = None
        self.private = None
        self.vehicle_type = None
        self.train_type = None
        self.date_from = None
        self.date_to = None
        self.report_key = None
    
    def set_params(self, params):
        self.query1 = params['query1']
        self.query2 = params['query2']
        # self.rail_company = params['rail_company']
        # self.station = params['station']
        self.county = params['county']
        self.state = params['state']
        self.city = params['city']
        self.highway = params['highway']
        self.private = params['private']
        # self.vehicle_type = params['vehicle_type']
        # self.train_type = params['train_type']
        self.date_from = params['date_from']
        self.date_to = params['date_to']
        self.report_key = params['report_key']

    def load_df_record_news(self):
        path_df_record_news = self.config_df_record_news.path
        if os.path.exists(path_df_record_news):
            self.df_record_news = pd.read_csv(path_df_record_news)
        else:
            self.df_record_news = pd.DataFrame(columns=self.config_df_record_news.columns)
            self.df_record_news.to_csv(path_df_record_news, index=False)
    
    def load_df_news_articles(self):
        path_df_news_articles = self.config_df_news_articles.path
        if os.path.exists(path_df_news_articles):
            self.df_news_articles = pd.read_csv(path_df_news_articles, parse_dates=['pub_date'])
        else:
            self.df_news_articles = pd.DataFrame(columns=self.config_df_news_articles.columns)
            self.df_news_articles.to_csv(path_df_news_articles, index=False)
    
    def append_df_record_news(self, df_temp):
        self.df_record_news = pd.concat([self.df_record_news, df_temp]).reset_index(drop=True)
        
    def save_df_record_news(self):
        path_df_record_news = self.config_df_record_news.path
        self.df_record_news.to_csv(path_df_record_news, index=False)
    
    def save_df_news_articles(self):
        path_df_news_articles = self.config_df_news_articles.path
        self.df_news_articles.to_csv(path_df_news_articles, index=False)
    
    def already_scraped(self):
        condition = (
            (self.df_record_news['report_key'] == self.report_key) &
            (self.df_record_news['query1'] == self.query1) &
            (self.df_record_news['query2'] == self.query2)
        )
        return condition.any()
    
    def get_RSS(self):
        # if self.private == 'Public':
            # query = f'{self.query1} {self.query2} {self.county} {self.city} {self.highway} after:{self.date_from} before:{self.date_to}'
        query = f'{self.query1} {self.query2} {self.county} {self.city} after:{self.date_from} before:{self.date_to}'
        params_rss = {
            "q": query,
            'hl': 'en-US',
            'gl': 'US',
            'ceid': 'US:en'
        }
        encoded_params_rss = urlencode(params_rss)
        feed_url = f'https://news.google.com/rss/search?{encoded_params_rss}'
        feed = feedparser.parse(feed_url) # type: ignore
        return feed
    
    def load_driver(self):
        platform_name = None
        os_name = platform.system()
        if os_name == "Linux":
            platform_name = 'linux64'
        elif os_name == "Darwin":
            arch = platform.machine()
            if arch == "x86_64":
                platform_name = 'mac-x64'
            elif arch == "arm64":
                platform_name = 'mac-arm64'
        else:
            print(f"Unsupported OS: {os_name}")
        
        self.quit_driver()
        options = Options()
        options.add_argument(
            "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/113.0.5672.63 Safari/537.36"
        )
        # options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1440,1080")
        options.add_argument("--incognito")
        if os_name == "Linux":
            options.add_argument("--headless")
            options.binary_location = f"./chrome-{platform_name}/chrome"
        chromedriver_path = f"./chromedriver-{platform_name}/chromedriver"
        service = ChromeService(executable_path=chromedriver_path)
        self.driver = webdriver.Chrome(options=options, service=service)
        self.driver.set_page_load_timeout(TIMEOUT) # if the page is not loaded in 10 seconds, an error occurs.
    
    def quit_driver(self):
        self.driver.quit()
        subprocess.call("pkill chromedriver", shell=True)
        subprocess.call("pkill chrome", shell=True)
    
    def screenshot_driver(self):
        self.driver.save_screenshot('web_screenshot.png')
    
    def get_article(self, feed):
        empty_row = [[self.query1, self.query2, self.county, self.state, self.city, self.highway, self.report_key] + ['']]
        df_result = pd.DataFrame(empty_row, columns=self.config_df_record_news.columns)
        pbar_entry = tqdm(feed['entries'], total=len(feed['entries']), leave=False)
        for entry in pbar_entry:
            # self.driver.delete_all_cookies()
            # _ = self.driver.execute_cdp_cmd("Network.clearBrowserCache", {})
            # _ = self.driver.execute_cdp_cmd("Network.clearBrowserCookies", {})

            try:
                news_id = entry['id']
                url = entry['link']
                pub_date = pd.to_datetime(entry['published'])
                title = entry['title']

                if news_id not in self.df_news_articles['news_id'].values:
                    redirect_url, page_source = self.get_redirect_url(url)
                    content_np_url, content_np_html = self.extract_content_newspaper3k(redirect_url, page_source)
                    content_tf_url, content_tf_html = self.extract_content_trafilatura(redirect_url, page_source)
                    content_rd_url, content_rd_html = self.extract_content_readability(redirect_url, page_source)
                    content_gs_url, content_gs_html = self.extract_content_goose3(redirect_url, page_source)

                    row_news_articles = [[news_id, redirect_url, pub_date, title, content_np_url, content_tf_url, content_rd_url, content_gs_url, content_np_html, content_tf_html, content_rd_html, content_gs_html]]
                    df_news_articles_temp = pd.DataFrame(row_news_articles, columns=self.config_df_news_articles.columns)
                    self.df_news_articles = pd.concat([self.df_news_articles, df_news_articles_temp])
                    self.save_df_news_articles()
                
                row_record_news = [[self.query1, self.query2, self.county, self.state, self.city, self.highway, self.report_key, news_id]]
                df_record_news_temp = pd.DataFrame(row_record_news, columns=self.config_df_record_news.columns)
                df_result = pd.concat([df_result, df_record_news_temp])
            except:
                self.quit_driver()
                self.load_driver()
        return df_result
    
    def get_redirect_url(self, url):
        self.driver.get(url)
        redirect_url = None
        if WebDriverWait(self.driver, TIMEOUT).until_not(EC.url_contains("google.com/rss/articles")) == False: # return value False means url doens't contain the pattern
            redirect_url = self.driver.current_url
        assert (redirect_url != None) and (redirect_url != url) and ("google.com/rss/articles" not in redirect_url)
        self.screenshot_driver()
        return redirect_url, self.driver.page_source
    
    def extract_content_newspaper3k(self, redirect_url, page_source):
        try:
            article = newspaper.Article(redirect_url, config=CONFIG_NP)
            article.download()
            article.parse()
            content_url = article.text
        except:
            content_url = None
        try:
            article = newspaper.Article('')
            article.set_html(page_source)
            article.parse()
            content_html = article.text
        except:
            content_html = None
        return content_url, content_html
    
    def extract_content_trafilatura(self, redirect_url, page_source):
        try:
            downloaded = trafilatura.fetch_url(redirect_url, config=CONFIG_TF)
            content_url = trafilatura.extract(downloaded, include_comments=False)
        except:
            content_url = None
        try:
            content_html = trafilatura.extract(page_source, include_comments=False)
        except:
            content_html = None
        return content_url, content_html
    
    def extract_content_readability(self, redirect_url, page_source):
        try:
            resp = requests.get(redirect_url, timeout=TIMEOUT)
            doc  = readability.Document(resp.text)
            content_url = BeautifulSoup(doc.summary(), "html.parser").get_text()
        except:
            content_url = None
        try:
            doc  = readability.Document(page_source)
            content_html = BeautifulSoup(doc.summary(), "html.parser").get_text()
        except:
            content_html = None
        return content_url, content_html
    
    def extract_content_goose3(self, redirect_url, page_source):
        g = goose3.Goose()
        try:
            g.config.http_timeout = TIMEOUT
            article = g.extract(redirect_url)
            content_url = article.cleaned_text
        except:
            content_url = None
        try:
            article = g.extract(raw_html=page_source)
            content_html = article.cleaned_text
        except:
            content_html = None
        return content_url, content_html
    
    # def press_and_hold(self):
    #     try:
    #         for _ in range(5):
    #             div = self.driver.find_element(By.ID, "px-captcha")
    #             x, y = div.size['width'], div.size['height']
    #             ActionChains(self.driver).move_to_element_with_offset(div, 0, -y * 0.25).click_and_hold().pause(5).release().perform()
    #         raise TimeoutError
    #     except:
    #         pass

def scrape_news(cfg: Config) -> tuple[pd.DataFrame | None, ...]:
    config_df_record_news = TableConfig(cfg.path.df_record_news, ['query1', 'query2', 'county', 'state', 'city', 'highway', 'report_key', 'news_id'])
    config_df_news_articles = TableConfig(cfg.path.df_news_articles, ['news_id', 'url', 'pub_date', 'title'] + list(cfg.scrp.news_crawlers))

    df_data: pd.DataFrame = pd.read_csv(cfg.path.df_record)
    df_data = df_data[df_data['State Name'].str.title().isin(cfg.scrp.target_states)]
    df_data['Date'] = pd.to_datetime(df_data['Date'])
    df_data = df_data[df_data['Date'] >= cfg.scrp.start_date]
    assert df_data['Report Key'].is_unique
    list_prior_info = ['Report Key', 'Railroad Name', 'Date', 'Nearest Station', 'County Name', 'State Name', 'City Name', 'Highway Name', 'Public/Private', 'Highway User', 'Equipment Type'] # keywords useful for searching
    df_data = df_data.sort_values(['County Name', 'Date'], ascending=[True, False])

    scrape = Scrape(config_df_record_news, config_df_news_articles)
    scrape.load_df_record_news()
    scrape.load_df_news_articles()

    # list_query1 = ["train", "amtrak", "locomotive"]
    # list_query2 = ["accident", "incident", "crash", "collide", "hit", "strike", "injure", "kill", "derail"]
    list_query1 = ["train"]
    list_query2 = ["accident"]

    pbar_row = tqdm(df_data.iterrows(), total=df_data.shape[0])
    for i, row in pbar_row:
        row = row.fillna('')
        report_key, rail_company, date, station, county, state, city, highway, private, vehicle_type, train_type = row[list_prior_info]
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
                    'date_from': (date - pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                    'date_to': (date + pd.Timedelta(days=7)).strftime('%Y-%m-%d'),
                    'report_key': report_key,
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
