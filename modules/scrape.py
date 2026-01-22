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

import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import os
import subprocess
import platform
import copy
import pathlib
from .config import Config, TableConfig
from .utils import (as_int, as_float, remove_dir, make_dir,
                    prepare_df_record, prepare_df_crossing, prepare_df_image, prepare_df_image_seq)
from pprint import pprint
from scipy.spatial.distance import cdist
import io
from PIL import Image
import py360convert
import math

TIMEOUT = 5
CONFIG_NP = newspaper.Config()
CONFIG_NP.request_timeout = TIMEOUT
CONFIG_TF = copy.deepcopy(trafilatura.settings.DEFAULT_CONFIG) # type: ignore
CONFIG_TF['DEFAULT']['DOWNLOAD_TIMEOUT'] = str(TIMEOUT)

class ScrapeNews:
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

    df_record = prepare_df_record(cfg)
    list_prior_info = ['Report Key', 'Railroad Name', 'Date', 'Nearest Station', 'County Name', 'State Name', 'City Name', 'Highway Name', 'Public/Private', 'Highway User', 'Equipment Type'] # keywords useful for searching
    df_record = df_record.sort_values(['County Name', 'Date'], ascending=[True, False])

    scrape = ScrapeNews(config_df_record_news, config_df_news_articles)
    scrape.load_df_record_news()
    scrape.load_df_news_articles()

    # list_query1 = ["train", "amtrak", "locomotive"]
    # list_query2 = ["accident", "incident", "crash", "collide", "hit", "strike", "injure", "kill", "derail"]
    list_query1 = ["train"]
    list_query2 = ["accident"]

    pbar_row = tqdm(df_record.iterrows(), total=df_record.shape[0])
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


class ScrapeImage:
    def __init__(self, cfg):
        self.cfg = cfg
        self.api_key = self.cfg.apikey.mapillary
        self.img_search_fields = ','.join(self.cfg.scrp.img_search_fields)
        self.img_detail_fields = ','.join(self.cfg.scrp.img_detail_fields)
        self.df_image = None
    
    def load_df_image(self):
        if os.path.exists(self.cfg.path.df_image):
            df_image = prepare_df_image(self.cfg)
        else:
            cols = list(self.cfg.scrp.img_detail_fields)
            cols.remove('geometry')
            cols.remove('computed_geometry')
            cols.extend(['lon', 'lat', 'computed_lon', 'computed_lat', 'dist', 'computed_dist'])
            cols = ['crossing'] + cols
            df_image = pd.DataFrame(columns=cols)
            df_image.to_csv(self.cfg.path.df_image, index=False)
        return df_image
    
    def load_df_image_seq(self):
        if os.path.exists(self.cfg.path.df_image_seq):
            df_image_seq = prepare_df_image_seq(self.cfg)
        else:
            cols = ['crossing_id', 'seq_id', 'img_pos', 'img_id', 'bearing']
            df_image_seq = pd.DataFrame(columns=cols)
            df_image_seq.to_csv(self.cfg.path.df_image_seq, index=False)
        return df_image_seq
    
    def search_images(self, bbox: str, limit: int = 10):
        """
        Query Mapillary for images inside a bounding box.
        Returns a list of image objects with basic metadata.
        """
        url = (
            "https://graph.mapillary.com/images"
            f"?access_token={self.api_key}"
            f"&bbox={bbox}"
            f"&fields={self.img_search_fields}"
            f"&limit={limit}"
        )

        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()

        # Mapillary returns {"data": [ ...images... ], ...maybe paging...}
        return data.get("data", [])

    def get_image_details(self, image_id: str):
        """
        Ask Mapillary for richer metadata for one specific image.
        Returns a dict with fields we asked for, including thumb_1024_url.
        """
        url = (
            f"https://graph.mapillary.com/{image_id}"
            f"?access_token={self.api_key}"
            f"&fields={self.img_detail_fields}"
        )

        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()

    def download_thumbnail(self, image_url: str, out_path: pathlib.Path):
        """
        Download the thumbnail JPG and save it locally.
        """
        resp = requests.get(image_url)
        resp.raise_for_status()

        out_path.write_bytes(resp.content)
        return out_path

    def get_image_seq(self, seq_id: str):
        """
        Query Mapillary for image sequence with a sequence key.
        Returns a list of image objects with basic metadata.
        """
        url = (
            'https://graph.mapillary.com/image_ids'
            f"?access_token={self.api_key}"
            f'&sequence_id={seq_id}'
        )

        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()
    
    def extract_view(self, e_img: np.ndarray, h_fov=90, yaw_deg=0, pitch_deg=0, out_hw=(512, 512)) -> np.ndarray:
        """
        Take a perspective view from the equirectangular pano.

        yaw_deg  (u_deg in py360convert): -left / +right (0 = forward)
        pitch_deg (v_deg): -down / +up
        fov_deg: horizontal FOV (or (h_fov, v_fov) tuple)
        out_hw: (height, width) of output image
        """
        aspect = out_hw[1] / out_hw[0]
        v_fov = 2 * math.degrees(math.atan(math.tan(math.radians(h_fov/2)) / aspect))
        pers = py360convert.e2p(
            e_img=e_img,
            fov_deg=(h_fov, v_fov),
            u_deg=yaw_deg,       # left/right
            v_deg=pitch_deg,     # up/down
            out_hw=out_hw,       # output size (H, W)
            in_rot_deg=0,
            mode="bilinear"
        )
        return pers

def scrape_image(cfg: Config) -> pd.DataFrame:
    df_crossing = prepare_df_crossing(cfg)
    scraper = ScrapeImage(cfg)
    df_image = scraper.load_df_image()
    for i, row in tqdm(df_crossing[['CROSSING', 'LATITUDE', 'LONGITUD']].iterrows(), total=df_crossing.shape[0]):
        crossing, lat, lon = row
        if crossing in df_image['crossing'].values:
            continue
        bbox_exact_match = f"{lon - cfg.scrp.bbox_offset},{lat - cfg.scrp.bbox_offset},{lon + cfg.scrp.bbox_offset},{lat + cfg.scrp.bbox_offset}"
        imgs = scraper.search_images(bbox_exact_match, limit=cfg.scrp.n_img)

        details_concat = [{'crossing': crossing}]
        for img in imgs:
            img_id = img["id"]
            if img_id in df_image['id'].values:
                continue
            details = scraper.get_image_details(img_id)
            if details.get('geometry', None):
                assert details['geometry']['type'] == 'Point'
                geometry = details.pop('geometry')
                details['lon'] = geometry['coordinates'][0]
                details['lat'] = geometry['coordinates'][1]
                dist = ((lat - details['lat'])**2 + (lon - details['lon'])**2)**0.5
                details['dist'] = dist
                assert dist <= cfg.scrp.bbox_offset * 2**0.5
                # if dist > cfg.scrp.bbox_offset:
                #     continue
            if details.get('computed_geometry', None):
                assert details['computed_geometry']['type'] == 'Point'
                computed_geometry = details.pop('computed_geometry')
                details['computed_lon'] = computed_geometry['coordinates'][0]
                details['computed_lat'] = computed_geometry['coordinates'][1]
                computed_dist = ((lon - details['computed_lon'])**2 + (lat - details['computed_lat'])**2)**0.5
                details['computed_dist'] = computed_dist
            if details.get('computed_rotation', None):
                assert isinstance(details['computed_rotation'], list)
            # print(f"computed:\t{computed_dist :.6f}")
            # print(f"actual:\t{dist :.6f}")
            # pprint(details, sort_dicts=False)

            details['crossing'] = crossing
            details_concat.append(details)
        
        df_image_temp = pd.DataFrame(details_concat, columns=df_image.columns)
        if not df_image_temp.empty:
            df_image = pd.concat([df_image, df_image_temp], ignore_index=True)
        
        if i % 10 == 0: # type: ignore
            df_image.to_csv(cfg.path.df_image, index=False)
        
    df_image.to_csv(cfg.path.df_image, index=False)

    for i, row in tqdm(df_image.iterrows(), total=df_image.shape[0]):
        img_id = as_int(row['id'])
        thumb_url = row["thumb_original_url"]
        if pd.isna(img_id):
            continue
        output_file = pathlib.Path(os.path.join(cfg.path.dir_scraped_images, f"{img_id}.jpg"))
        if not output_file.exists() and pd.notna(thumb_url):
            scraper.download_thumbnail(thumb_url, output_file)

    return df_image

def scrape_image_seq(cfg: Config) -> pd.DataFrame:
    scraper = ScrapeImage(cfg)
    df_crossing = prepare_df_crossing(cfg)
    df_image = scraper.load_df_image()
    df_image_seq = scraper.load_df_image_seq()
    
    ############### using only actual GPS & highway-xing
    df_crossing = df_crossing[df_crossing['CROSSING'].isin(df_image[df_image['id'].notna()]['crossing'].unique())]
    df_crossing = df_crossing[df_crossing['LLSOURCE'].isin(['1'])] # ['1', '2', ' '];  ' ' mostly incorrect, '2' sometimes incorrect
    df_crossing = df_crossing[df_crossing['XPURPOSE'] == 1] # 1: highway, 2: pedestrian pathway, 3: train station / [2,3] images are generally not available
    df_crossing = df_crossing.drop(['HIGHWAY', 'RRDIV', 'RRSUBDIV'], axis=1)
    # print(df_crossing[df_crossing['STREET'].str.lower().str.contains('wright')])

    ############### using only pano
    df_image = df_image.dropna(subset=['id'])
    df_image['id'] = df_image['id'].astype(int)
    df_image = df_image[df_image['is_pano'] == 1]
    df_image = df_image[df_image['camera_type'] == 'spherical'] # [spherical, equirectangular] equirectangular images require diff view extraction mechanism
    
    ############### merge
    df_image = df_image.merge(df_crossing[['CROSSING', 'LATITUDE', 'LONGITUD', 'STREET']], left_on='crossing', right_on='CROSSING')
    df_image = df_image.drop(columns=['CROSSING'])
    
    ############### using only images with distance over the threshold
    # df_image = df_image[(df_image['dist'] <= threshold) | (df_image['computed_dist'] <= threshold)]
    df_min_dist = df_image.loc[df_image.groupby("crossing")["dist"].idxmin()][['crossing', 'id', 'compass_angle', 'computed_compass_angle', 'sequence', 'lat', 'lon', 'LATITUDE', 'LONGITUD', 'dist']].reset_index(drop=True)
    df_min_dist = df_min_dist[df_min_dist['dist'] <= cfg.scrp.dist_thresh_from_crossing] # it seems actual GPS location is more accurate than computed GPS location, so I only use `dist` here, not `computed_dist`.
    # print(df_min_dist)
    
    ############### get image seq
    (((df_image['lat'] - df_image['computed_lat'])**2 + (df_image['lon'] - df_image['computed_lon'])**2)**0.5).dropna().sort_values()
    for i, row in tqdm(df_min_dist.iterrows(), total=df_min_dist.shape[0]):
        crossing_id = row['crossing']
        if crossing_id in df_image_seq['crossing_id'].values:
            continue
        seq_id = row['sequence']
        xing_lat, xing_lon = row[['LATITUDE', 'LONGITUD']]
        
        df_seq_temp = df_image[(df_image['crossing'] == crossing_id) & (df_image['sequence'] == seq_id)]
        df_seq_temp = df_seq_temp[(df_seq_temp['computed_compass_angle'].notna())]
        # select_img_within = [0.0001, 0.0002]
        # df_seq_temp = df_seq_temp[(select_img_within[0] <= df_seq_temp['dist']) & (df_seq_temp['dist'] <= select_img_within[1])]
        
        if df_seq_temp.shape[0] <= 1:
            continue
        
        images = scraper.get_image_seq(seq_id)['data']
        images = [image['id'] for image in images]
        # for image in tqdm(images, leave=False):
        #     img_id = image['id']
        #     details = scraper.get_image_details(img_id)
        #     dist = ((row[['LONGITUD', 'LATITUDE']] - details['geometry']['coordinates'])**2).sum()**0.5
        #     if dist > cfg.scrp.bbox_offset * 2 or dist < cfg.scrp.bbox_offset / 2:
        #         continue
        #     assert details['geometry']['type'] == 'Point'
        #     geometry = details.pop('geometry')
        #     details['lon'] = geometry['coordinates'][0]
        #     details['lat'] = geometry['coordinates'][1]
        
        # print(df_seq_temp)
        df_image_seq_temp = pd.DataFrame(columns=df_image_seq.columns)
        for _, seq_row in df_seq_temp.iterrows():
            img_id = seq_row['id']
            if str(img_id) not in images or pd.isna(seq_row['computed_compass_angle']):
                continue
            img_pos = str(images.index(str(img_id))).zfill(4)
            lat, lon = seq_row[['lat', 'lon']]
            # camera_yaw_deg = seq_row['compass_angle'] # from the raw GPS trajectory possibly with error away from the actual roadway
            camera_yaw_deg = seq_row['computed_compass_angle'] # corrected version according to the actual roadway
            # _, camera_pitch_deg, camera_yaw_deg = list(map(lambda d: int((360 + math.degrees(d)) % 360), seq_row['computed_rotation'])) # from rotation [roll, pitch, yaw]

            lat1, lon1, lat2, lon2 = map(math.radians, [lat, lon, xing_lat, xing_lon])
            d_lon = lon2 - lon1
            x = math.sin(d_lon) * math.cos(lat2)
            y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
            bearing = math.degrees(math.atan2(x, y))
            bearing = int((bearing - camera_yaw_deg + 360) % 360)
            
            fp_img = os.path.join(cfg.path.dir_scraped_images, f'{img_id}.jpg')
            if not os.path.exists(fp_img):
                continue
            img = np.array(Image.open(fp_img).convert("RGB"))
            view = scraper.extract_view(img, h_fov=90, yaw_deg=bearing, pitch_deg=0, out_hw=(720, 960))
            out_img = Image.fromarray(view)
            dp_crossing_seq = os.path.join(cfg.path.dir_image_seq, seq_id)
            make_dir(dp_crossing_seq)
            fp_img_seq = os.path.join(dp_crossing_seq, f'{img_pos}_{img_id}.jpg')
            if not os.path.exists(fp_img_seq):
                out_img.save(fp_img_seq)

            df_image_seq_temp.loc[len(df_image_seq_temp)] = [crossing_id, seq_id, img_pos, img_id, bearing]
        
        if not df_image_seq_temp.empty:
            df_image_seq = pd.concat([df_image_seq, df_image_seq_temp])
        
        if i % 10 == 0: # type: ignore
            df_image_seq.to_csv(cfg.path.df_image_seq, index=False)

    df_image_seq.to_csv(cfg.path.df_image_seq, index=False)
    return df_image_seq

if __name__ == '__main__':
    ### test
    
    # img_id = 2726828097607512
    # img_id = 1125481356095921
    # img_id = 375830210537976
    # df_image[df_image['id'] == img_id].iloc[0][['compass_angle', 'computed_compass_angle']]
    # roll, pitch, yaw = df_image[df_image['id'] == img_id].iloc[0]['computed_rotation'] # x(tilt r/l), y(look u/d), z(compass) in 3D (default is east, down, north)
    # roll_deg, pitch_deg, yaw_deg = list(map(math.degrees, [roll, pitch, yaw]))
    # roll_deg, pitch_deg, yaw_deg

    # remove_dir(cfg.path.dir_scraped_images)
    # make_dir(cfg.path.dir_scraped_images)
    # df_crossing[df_crossing['STREET'].str.upper().str.contains('SPRUCE')]
    # df_crossing[df_crossing['STREET'].str.upper().str.contains('CHICAGO')]
    # df_crossing[df_crossing['STREET'].str.upper().str.contains('IOWA AVE')]
    # df_crossing[df_crossing['STREET'].str.upper().str.contains('PALM AVE')]
    # df_crossing[df_crossing['STREET'].str.upper().str.contains('BROCKTON AVE')]
    # df_crossing[df_crossing['STREET'].str.upper().str.contains('WASHINGTON')]
    lat, lon = 33.990282, -117.356688 # SPRUCE ST 0.00002
    lat, lon = 33.997812, -117.348500 # CHICAGO AVE 0.00004
    lat, lon = 33.9942, -117.339849 # IOWA AVE 0.00007
    lat, lon = 33.957269, -117.396787 # BROCKTON AVE 0.00004
    lat, lon = 33.957246, -117.401092 # PALM AVE 0.00003
    lat, lon = 33.938530, -117.396230 # WASHINGTON AVE 0.00003