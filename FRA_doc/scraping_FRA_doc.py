from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import pandas as pd
import itertools
from tqdm import tqdm
import time
import random
import os

DATA_FOLDER = 'data/'
FRA_DOC_FOLDER = 'doc_FRA/'
TIME_WAIT = 2

def get_author(driver):
    try:
        list_div_author_name = WebDriverWait(driver, TIME_WAIT).until(
            EC.visibility_of_all_elements_located((By.XPATH, './/div[@class="field field--name-field-author field--type-string field--label-inline"]/div[@class="field__items"]/div'))
        )
        author = '||'.join([div_author_name.text for div_author_name in list_div_author_name])
    except:
        author = float('nan')
    return author
    
def get_paragraph(driver):
    try:
        dict_paragraph = {}
        list_div_subparagraph = WebDriverWait(driver, TIME_WAIT).until(
            EC.visibility_of_all_elements_located((By.XPATH, '//div[@class="field field--name-field-document-type field--type-entity-reference-revisions field--label-hidden"]/div/div/div/div'))
        )
        for div_subparagraph in list_div_subparagraph:
            label_subparagraph = WebDriverWait(div_subparagraph, TIME_WAIT).until(
                EC.visibility_of_element_located((By.CLASS_NAME, 'field__label'))
            ).text
            list_item_subparagraph = WebDriverWait(div_subparagraph, TIME_WAIT).until(
                EC.visibility_of_all_elements_located((By.CLASS_NAME, 'field__item'))
            )
            item_subparagraph = '||'.join([item.text for item in list_item_subparagraph])
            dict_paragraph[label_subparagraph] = item_subparagraph
        dict_paragraph = str(dict_paragraph)
    except:
        dict_paragraph = float('nan')
    return dict_paragraph

def get_keyword(driver):
    try:
        keyword_loc = WebDriverWait(driver, TIME_WAIT).until(
            EC.visibility_of_element_located((By.XPATH, './/div[@class="field field--name-field-keywords field--type-string field--label-inline"]/div[@class="field__item"]'))
        )
        keyword = keyword_loc.text
    except:
        keyword = float('nan')
    return keyword

options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1000x2000")
options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36")
# options.add_argument("--disable-application-cache")
# options.add_argument("--disable-infobars")
# options.add_argument("--no-sandbox")
# options.add_argument('--disable-dev-shm-usage')
# options.add_argument("--hide-scrollbars")
# options.add_argument("--enable-logging")
# options.add_argument("--log-level=0")
# options.add_argument("--single-process")
# options.add_argument("--ignore-certificate-errors")
# options.add_argument("--homedir=./TMP")
prefs = {"download.default_directory" : f"./{DATA_FOLDER}{FRA_DOC_FOLDER}"}
options.add_experimental_option("prefs", prefs)
options.binary_location = "./chrome-linux64/chrome"
chromedriver_path = "./chromedriver-linux64/chromedriver"
service = ChromeService(executable_path=chromedriver_path)

################################# SCRAPE ALL DOC ########################################
if not os.path.exists(DATA_FOLDER + FRA_DOC_FOLDER):
    os.mkdir(DATA_FOLDER + FRA_DOC_FOLDER)
df_FRA_doc_url = pd.read_csv(DATA_FOLDER + 'df_FRA_doc_url.csv')
df_FRA_doc_url['date'] = pd.to_datetime(df_FRA_doc_url['date'])

if os.path.exists(DATA_FOLDER + 'df_FRA_doc.csv'):
    df_FRA_doc = pd.read_csv(DATA_FOLDER + 'df_FRA_doc.csv')
    df_FRA_doc['date'] = pd.to_datetime(df_FRA_doc['date'])
else:
    df_FRA_doc = pd.DataFrame(columns=['series', 'subseries', 'date', 'url', 'pdf', 'author', 'description', 'keyword'])
driver = webdriver.Chrome(options=options, service=service)
for i, row in tqdm(df_FRA_doc_url.iterrows(), total=df_FRA_doc_url.shape[0]):
    series, subseries, date, url, _ = row
    if (df_FRA_doc['url'] == url).any():
        continue
    driver.get(url)
    _ = driver.save_screenshot('web_screenshot.png')
    
    author = get_author(driver)
    dict_paragraph = get_paragraph(driver)
    keyword = get_keyword(driver)
    
    try:
        file = WebDriverWait(driver, TIME_WAIT).until(
            EC.element_to_be_clickable((By.XPATH, './/a[@type="application/pdf"]'))
        )
        if not os.path.exists(DATA_FOLDER + FRA_DOC_FOLDER + file.text):
            file.click()
        pdf = True
    except:
        pdf = False
    df_temp = pd.DataFrame({
        'series': [series], 'subseries': [subseries], 'date': [date], 'url': [url], 'pdf': [pdf], 
        'author': [author], 'description': [dict_paragraph], 'keyword': [keyword]
    })
    df_FRA_doc = pd.concat([df_FRA_doc, df_temp]).reset_index(drop=True)
    df_FRA_doc.to_csv(DATA_FOLDER + 'df_FRA_doc.csv', index=False)
driver.close()