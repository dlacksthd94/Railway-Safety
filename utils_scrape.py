from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.action_chains import ActionChains

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
import hashlib

TIMEOUT = 5
CONFIG_NP = newspaper.Config()
CONFIG_NP.request_timeout = TIMEOUT
CONFIG_TF = copy.deepcopy(trafilatura.settings.DEFAULT_CONFIG)
CONFIG_TF['DEFAULT']['DOWNLOAD_TIMEOUT'] = str(TIMEOUT)

class Scrape():
    def __init__(self, columns):
        self.driver = None
        self.df = None
        self.columns = columns
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
        self.incident_id = None
    
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
        self.incident_id = params['incident_id']

    def load_df(self, path_df_cal):
        if os.path.exists(path_df_cal):
            self.df = pd.read_csv(path_df_cal, parse_dates=['pub_date'])
        else:
            self.df = pd.DataFrame(columns=self.columns)
            self.df.to_csv(path_df_cal, index=False)

    def append_df(self, df_temp):
        self.df = pd.concat([self.df, df_temp]).reset_index(drop=True)

    def save_df(self, path_df_cal):
        self.df.to_csv(path_df_cal, index=False)
    
    def already_scraped(self):
        condition = (
            (self.df['incident_id'] == self.incident_id) &
            (self.df['query1'] == self.query1) &
            (self.df['query2'] == self.query2)
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
        feed = feedparser.parse(feed_url)
        return feed
    
    def load_driver(self):
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
        
        if self.driver is not None:
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
        empty_row = [[self.query1, self.query2, self.county, self.state, self.city, self.highway, self.incident_id] + ['', '', '', ''] + [''] * 8]
        df_result = pd.DataFrame(empty_row, columns=self.columns)
        pbar_entry = tqdm(feed['entries'], total=len(feed['entries']), leave=False)
        for entry in pbar_entry:
            self.driver.delete_all_cookies()
            _ = self.driver.execute_cdp_cmd("Network.clearBrowserCache", {})
            _ = self.driver.execute_cdp_cmd("Network.clearBrowserCookies", {})

            try:
                news_id = entry['id']
                url = entry['link']
                pub_date = pd.to_datetime(entry['published'])
                title = entry['title']
                if news_id in self.df['news_id'].values:
                    continue
                redirect_url, page_source = self.get_redirect_url(url)
                content_np_url, content_np_html = self.extract_content_newspaper3k(redirect_url, page_source)
                content_tf_url, content_tf_html = self.extract_content_trafilatura(redirect_url, page_source)
                content_rd_url, content_rd_html = self.extract_content_readability(redirect_url, page_source)
                content_gs_url, content_gs_html = self.extract_content_goose3(redirect_url, page_source)
                row_data = [[self.query1, self.query2, self.county, self.state, self.city, self.highway, self.incident_id, news_id, redirect_url, pub_date, title, content_np_url, content_np_html, content_tf_url, content_tf_html, content_rd_url, content_rd_html, content_gs_url, content_gs_html]]
                df_temp = pd.DataFrame(row_data, columns=self.columns)
                df_result = pd.concat([df_result, df_temp])
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
            content_html
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
    
    def press_and_hold(self):
        try:
            for _ in range(5):
                div = self.driver.find_element(By.ID, "px-captcha")
                x, y = div.size['width'], div.size['height']
                ActionChains(self.driver).move_to_element_with_offset(div, 0, -y * 0.25).click_and_hold().pause(5).release().perform()
            raise TimeoutError
        except:
            pass

def hash_row(row):
    strs = row.astype(str).to_list()
    blob = ''.join(strs).encode('utf-8')
    h = hashlib.md5(blob).hexdigest()
    return h

if __name__ == "__main__":
    self = scrape
    it = iter(feed['entries'])
    entry = next(it)
    entry = next(it)
    entry = next(it)
    entry = next(it)