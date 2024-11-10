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
TIME_WAIT = 5

def find_next_button(driver):
    try:
        # Wait for the element to become clickable
        next_button = WebDriverWait(driver, TIME_WAIT).until(
            EC.element_to_be_clickable((By.ID, 'pnnext'))
        )
        return next_button  # Return the button if found
    except TimeoutException:
        return 0

def article_exists(driver):
    try:
        _ = WebDriverWait(driver, TIME_WAIT).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'mnr-c'))
        )
        return True  # Element found
    except TimeoutException:
        return False

def is_captcha(driver):
    try:
        _ = WebDriverWait(driver, 1).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="recaptcha"]'))
        )
        return True  # Element found
    except TimeoutException:
        return False


options = Options()
options.add_argument("--headless")
# options.add_argument("--disable-gpu")
options.add_argument("--window-size=1000x2000")
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
options.binary_location = "./chromedriver/chrome"
chromedriver_path = "./chromedriver/chromedriver"
service = ChromeService(executable_path=chromedriver_path)

################################# SEARCH BY STATE ########################################
if os.path.exists(DATA_FOLDER + 'df_url_state.csv'):
    df_url_state = pd.read_csv(DATA_FOLDER + 'df_url_state.csv')
else:
    df_url_state = pd.DataFrame(columns=['subject', 'action', 'spot', 'state', 'year', 'page', 'url'])

list_subject = ['train']
# list_action = ['crash', 'collision', 'pileup', 'accident', 'smash', 'run into', 'struck', 'hit']
list_action = ['crash']
# list_spot = ['crossing', 'intersection']
list_spot = ['crossing']
list_state = ['MISSISSIPPI', 'INDIANA', 'FLORIDA', 'GEORGIA', 'KENTUCKY', 'OKLAHOMA', 'ILLINOIS', 'MICHIGAN', 'MINNESOTA', 'SOUTH CAROLINA', 'NORTH CAROLINA', 'IOWA', 'MISSOURI', 'MAINE', 'MARYLAND', 'MASSACHUSETTS', 'LOUISIANA', 'NEW JERSEY', 'WEST VIRGINIA', 'VIRGINIA', 'OHIO', 'WISCONSIN', 'NEW YORK', 'KANSAS', 'PENNSYLVANIA', 'MONTANA', 'NORTH DAKOTA', 'CALIFORNIA', 'ALASKA', 'TENNESSEE', 'ARKANSAS', 'ALABAMA', 'TEXAS', 'WASHINGTON', 'IDAHO', 'SOUTH DAKOTA', 'NEBRASKA', 'UTAH', 'COLORADO', 'WYOMING', 'ARIZONA', 'NEW MEXICO', 'NEW HAMPSHIRE', 'OREGON', 'VERMONT', 'RHODE ISLAND', 'DELAWARE', 'CONNECTICUT', 'DISTRICT OF COLUMBIA', 'NEVADA', 'HAWAII']
list_year = range(2010, 2025)
list_search_keyword = list(itertools.product(list_subject, list_action, list_spot, list_state, list_year))

while not ((df_url_state['state'] == list_state[-1]) & (df_url_state['year'] == list_year[-1])).any():
    try:
        for i, (subject, action, spot, state, year) in enumerate(tqdm(list_search_keyword)):
            print(i, subject, action, spot, state, year)
            already_scraped = ((df_url_state['subject'] == subject) & (df_url_state['action'] == action) & (df_url_state['spot'] == spot) & (df_url_state['state'] == state) & (df_url_state['year'] == year)).any()
            if already_scraped:
                continue
            driver = webdriver.Chrome(options=options, service=service)
            driver.get('https://www.croxyproxy.com/')
            url = f'https://www.google.com/search?q={subject}+{action}+{spot}+%22{"+".join(state.split())}%22&lr=&cr=countryUS&sca_esv=992c006e90961474&as_qdr=all&biw=2752&bih=991&sxsrf=ADLYWILM8AXcEYUEQl4yA1oQLkPzMDzQLA%3A1729122001018&source=lnt&tbs=ctr%3AcountryUS%2Ccdr%3A1%2Ccd_min%3A{year}%2Ccd_max%3A{year}&tbm=nws'
            search_bar = WebDriverWait(driver, TIME_WAIT).until(
                EC.element_to_be_clickable((By.ID, 'url'))
            )
            search_bar.send_keys(url, Keys.ENTER)
            # while is_captcha(driver=driver):
            #     time.sleep(10)
            # time.sleep(random.choice(range(1, 5)))
            # _ = driver.save_screenshot('web_screenshot.png')
            list_url = []
            num_page = 0
            while True:
                # check if therer is any articles
                result_empty = article_exists(driver=driver)
                _ = driver.save_screenshot('web_screenshot.png')
                if result_empty:
                    break
                num_page += 1
                print(num_page)
                list_article_element = WebDriverWait(driver, TIME_WAIT).until(
                    EC.visibility_of_all_elements_located((By.CLASS_NAME, 'WlydOe'))
                )
                list_url_temp = list(map(lambda e: e.get_attribute('href'), list_article_element))
                list_url.append(list_url_temp)
                next_button = find_next_button(driver=driver)
                if next_button:
                    next_button.click()
                    # time.sleep(random.choice(range(1, 5)))
                    # _ = driver.save_screenshot('web_screenshot.png')
                else:
                    break
            df_url_state_temp = pd.DataFrame(columns=['subject', 'action', 'spot', 'state', 'year', 'page', 'url'])
            if len(list_url) == 0:
                df_url_state_temp.loc[0] = [subject, action, spot, state, year, 1, float('nan')]
            else:
                df_url_state_temp[['subject', 'action', 'spot', 'state', 'year', 'page']] = [[subject, action, spot, state, year, i] for i in range(1, num_page + 1)]
                df_url_state_temp['url'] = list_url
                df_url_state_temp = df_url_state_temp.explode('url')
            df_url_state = pd.concat([df_url_state, df_url_state_temp]).reset_index(drop=True)
            df_url_state.to_csv(DATA_FOLDER + 'df_url_state.csv', index=False)
            driver.close()
    except:
        pass

####################################### SEARCH BY DATE #######################################
if os.path.exists(DATA_FOLDER + 'df_url_date.csv'):
    df_url_date = pd.read_csv(DATA_FOLDER + 'df_url_date.csv')
    df_url_date['date'] = pd.to_datetime(df_url_date['date'])
else:
    df_url_date = pd.DataFrame(columns=['subject', 'action', 'spot', 'date', 'page', 'url'])

list_subject = ['train']
# list_action = ['crash', 'collision', 'pileup', 'accident', 'smash', 'run into', 'struck', 'hit']
list_action = ['crash']
# list_spot = ['crossing', 'intersection']
list_spot = ['crossing']
list_date = pd.date_range('2010-01-01', '2024-12-31')
list_search_keyword = list(itertools.product(list_subject, list_action, list_spot, list_date))

while not (df_url_date['date'] == list_date[-1]).any():
    try:
        for i, (subject, action, spot, date) in enumerate(tqdm(list_search_keyword)):
            print(i, subject, action, spot, date)
            already_scraped = ((df_url_date['subject'] == subject) & (df_url_date['action'] == action) & (df_url_date['spot'] == spot) & (df_url_date['date'] == date)).any()
            if already_scraped:
                continue
            driver = webdriver.Chrome(options=options, service=service)
            driver.get('https://www.croxyproxy.com/')
            url = f'https://www.google.com/search?q={subject}+{action}+{spot}&lr=&cr=countryUS&sca_esv=992c006e90961474&biw=2752&bih=991&sxsrf=ADLYWILwzvuhE4yOKhwNuLF9JIEt7l-pIg%3A1729582927880&source=lnt&tbs=ctr%3AcountryUS%2Ccdr%3A1%2Ccd_min%3A{date.month}%2F{date.day}%2F{date.year}%2Ccd_max%3A{date.month}%2F{date.day}%2F{date.year}&tbm=nws'
            search_bar = WebDriverWait(driver, TIME_WAIT).until(
                EC.element_to_be_clickable((By.ID, 'url'))
            )
            search_bar.send_keys(url, Keys.ENTER)
            # _ = driver.save_screenshot('web_screenshot.png')
            list_url = []
            num_page = 0
            while True:
                # check if therer is any articles
                result_empty = article_exists(driver=driver)
                _ = driver.save_screenshot('web_screenshot.png')
                if result_empty:
                    break
                num_page += 1
                print(num_page)
                list_article_element = WebDriverWait(driver, TIME_WAIT).until(
                    EC.visibility_of_all_elements_located((By.CLASS_NAME, 'WlydOe'))
                )
                list_url_temp = list(map(lambda e: e.get_attribute('href'), list_article_element))
                list_url.append(list_url_temp)
                next_button = find_next_button(driver=driver)
                if next_button:
                    next_button.click()
                    # time.sleep(random.choice(range(1, 5)))
                    # _ = driver.save_screenshot('web_screenshot.png')
                else:
                    break
            df_url_date_temp = pd.DataFrame(columns=['subject', 'action', 'spot', 'date', 'page', 'url'])
            if len(list_url) == 0:
                df_url_date_temp.loc[0] = [subject, action, spot, date, 1, float('nan')]
            else:
                df_url_date_temp[['subject', 'action', 'spot', 'date', 'page']] = [[subject, action, spot, date, i] for i in range(1, num_page + 1)]
                df_url_date_temp['url'] = list_url
                df_url_date_temp = df_url_date_temp.explode('url')
            df_url_date = pd.concat([df_url_date, df_url_date_temp]).reset_index(drop=True)
            df_url_date.to_csv(DATA_FOLDER + 'df_url_date.csv', index=False)
            driver.close()
    except:
        pass