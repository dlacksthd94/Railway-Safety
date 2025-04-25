from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

import feedparser
from urllib.parse import urlencode
from newspaper import Article
import trafilatura
from readability import Document
import requests
from bs4 import BeautifulSoup
from goose3 import Goose

import pandas as pd
import time
from tqdm import tqdm
import os
import subprocess
import platform
from selenium.webdriver.common.action_chains import ActionChains

class Scrape():
    def __init__(self, columns):
        self.driver = None
        self.df = None
        self.columns = columns
        self.query1 = None
        self.query2 = None
        self.state = None
        self.county = None
        self.city = None
        self.date_from = None
        self.date_to = None
    
    def set_params(self, params):
        self.query1 = params['query1']
        self.query2 = params['query2']
        self.state = params['state']
        self.county = params['county']
        self.city = params['city']
        self.date_from = params['date_from']
        self.date_to = params['date_to']

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
            (self.df['query1'] == self.query1) &
            (self.df['query2'] == self.query2) &
            (self.df['state'] == self.state) &
            (self.df['county'] == self.county) & 
            (self.df['city'] == self.city)
        )
        return condition.any()
    
    def get_RSS(self):
        params_rss = {
            "q": f'{self.query1} {self.query2} state:{self.state} county:{self.county} city:{self.city} after:{self.date_from} before:{self.date_to}',
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
        self.driver.set_page_load_timeout(10) # if the page is not loaded in 10 seconds, an error occurs.
    
    def quit_driver(self):
        self.driver.quit()
        subprocess.call("pkill chromedriver", shell=True)
        subprocess.call("pkill chrome", shell=True)
    
    def screenshot_driver(self):
        self.driver.save_screenshot('web_screenshot.png')
    
    def get_article(self, feed):
        empty_row = [[self.query1, self.query2, self.state, self.county, self.city, '', '', '', '', '', '', '', '', '']]
        df_result = pd.DataFrame(empty_row, columns=self.columns)
        pbar_entry = tqdm(feed['entries'], leave=False)
        for entry in pbar_entry:
            self.driver.delete_all_cookies()
            _ = self.driver.execute_cdp_cmd("Network.clearBrowserCache", {})
            _ = self.driver.execute_cdp_cmd("Network.clearBrowserCookies", {})

            try:
                id = entry['id']
                url = entry['link']
                pub_date = pd.to_datetime(entry['published'])
                title = entry['title']
                if id in self.df['id'].values:
                    continue
                redirect_url, content = self.get_redirect_url(url)
                content_newspaper3k = self.extract_content_newspaper3k(redirect_url)
                content_trafilatura = self.extract_content_trafilatura(redirect_url)
                content_readability = self.extract_content_readability(redirect_url)
                content_goose3 = self.extract_content_goose3(redirect_url)
                row_data = [[self.query1, self.query2, self.state, self.county, self.city, id, redirect_url, pub_date, title, content, content_newspaper3k, content_trafilatura, content_readability, content_goose3]]
                df_temp = pd.DataFrame(row_data, columns=self.columns)
                df_result = pd.concat([df_result, df_temp])
            except:
                self.quit_driver()
                self.load_driver()
        return df_result
    
    def get_redirect_url(self, url):
        self.driver.get(url)
        redirect_url = None
        if WebDriverWait(self.driver, 10).until_not(EC.url_contains("google.com/rss/articles")) == False: # return value False means url doens't contain the pattern
            redirect_url = self.driver.current_url
        assert (redirect_url != None) and (redirect_url != url) and ("google.com/rss/articles" not in redirect_url)
        self.screenshot_driver()
        
        article = Article('')
        article.set_html(self.driver.page_source)
        article.parse()
        text = article.text
        return redirect_url, text
    
    def extract_content_newspaper3k(self, redirect_url):
        try:
            article = Article(redirect_url)
            article.download()
            article.parse()
            text = article.text
        except:
            text = None
        return text
    
    def extract_content_trafilatura(self, redirect_url):
        try:
            downloaded = trafilatura.fetch_url(redirect_url)
            text = trafilatura.extract(downloaded, include_comments=False)
        except:
            text = None
        return text
    
    def extract_content_readability(self, redirect_url):
        try:
            resp = requests.get(redirect_url)
            doc  = Document(resp.text)
            text = BeautifulSoup(doc.summary(), "html.parser").get_text()
        except:
            text = None
        return text
    
    def extract_content_goose3(self, redirect_url):
        try:
            g = Goose()
            article = g.extract(redirect_url)
            text = article.cleaned_text
        except:
            text = None
        return text
    
    def press_and_hold(self):
        try:
            for _ in range(5):
                div = self.driver.find_element(By.ID, "px-captcha")
                x, y = div.size['width'], div.size['height']
                ActionChains(self.driver).move_to_element_with_offset(div, 0, -y * 0.25).click_and_hold().pause(5).release().perform()
            raise TimeoutError
        except:
            pass

class VerificationError(Exception):
    pass


if __name__ == "__main__":
    self = scrape
    it = iter(feed['entries'])
    entry = next(it)
    entry = next(it)
    entry = next(it)
    entry = next(it)