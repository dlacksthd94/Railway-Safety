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
TIME_WAIT = 5
START_DATE = "2000-01-01"

def last_button_clicked(driver):
    try:
        a_last_button = WebDriverWait(driver, TIME_WAIT).until(
            EC.element_to_be_clickable((By.XPATH, './/a[@title="Go to last page"]'))
        )
        a_last_button.click()
        return True
    except:
        return False

def prev_button_clicked(driver):
    a_prev_button = WebDriverWait(driver, TIME_WAIT).until(
        EC.element_to_be_clickable((By.XPATH, './/a[@title="Go to previous page"]'))
    )
    a_prev_button.click()
    return True

def get_page_num(driver):
    # Wait for the element to become clickable
    try:
        li_page_num = WebDriverWait(driver, TIME_WAIT).until(
            EC.element_to_be_clickable((By.XPATH, './/li[@class="page-item active"]/span'))
        )
        page_num = int(li_page_num.text)
        return page_num
    except:
        return 1

def find_docs(driver):
    list_row = WebDriverWait(driver, TIME_WAIT).until(
        EC.visibility_of_all_elements_located((By.CLASS_NAME, 'search-item.elibrary--search-item.views-row'))
    )
    return list_row  # Element found


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
options.binary_location = "./chromedriver/chrome"
chromedriver_path = "./chromedriver/chromedriver"
service = ChromeService(executable_path=chromedriver_path)

############################# GATHER DOCUMENT CATEGORY ##################################
driver = webdriver.Chrome(options=options, service=service)
driver.get(f'https://railroads.dot.gov/elibrary-search?')
time.sleep(TIME_WAIT)
button_document_type = driver.find_element(By.ID, '#collapse-documentseries')
button_document_type.click()
assert button_document_type.get_attribute('aria-expanded') == 'true'

list_label_series = WebDriverWait(driver, TIME_WAIT).until(
    EC.presence_of_all_elements_located((By.XPATH, './/ul[@data-drupal-facet-id="document_series"]/li/label'))
)
dict_series_id = {}
for label_series in list_label_series:
    id_series = label_series.get_attribute('for').split('-')[-1]
    series = label_series.find_element(By.CLASS_NAME, 'facet-item__value').text
    count_series = int(label_series.find_element(By.CLASS_NAME, 'facet-item__count').text.strip('()'))
    dict_series_id[series] = {'id': id_series, 'subseries': {}, 'count': count_series}

url_series_unfolded = f'https://railroads.dot.gov/elibrary-search?'
for i, (key, value) in enumerate(dict_series_id.items()):
    id_series = value['id']
    url_series_unfolded += f'&f%5B{i}%5D=document_series%3A{id_series}'
driver.get(url_series_unfolded)
time.sleep(TIME_WAIT)

list_label_series = WebDriverWait(driver, TIME_WAIT).until(
    EC.presence_of_all_elements_located((By.XPATH, './/ul[@data-drupal-facet-id="document_series"]/li'))
)
for label_series in list_label_series:
    list_label_subseries = label_series.find_elements(By.XPATH, './/ul[@class="accordion list-unstyled py-3"]/li/label')
    series = label_series.find_element(By.CLASS_NAME, 'facet-item__value').text
    for label_subseries in list_label_subseries:
        id_subseries = label_subseries.get_attribute('for').split('-')[-1]
        subseries = label_subseries.find_element(By.CLASS_NAME, 'facet-item__value').text
        count_subseries = int(label_series.find_element(By.CLASS_NAME, 'facet-item__count').text.strip('()'))
        dict_series_id[series]['subseries'][subseries] = {'id': id_subseries, 'count': count_subseries}
dict_series_id
driver.close()

################################# SCRAPE DOC URL ########################################
if os.path.exists(DATA_FOLDER + 'df_FRA_doc_url.csv'):
    df_FRA_doc_url = pd.read_csv(DATA_FOLDER + 'df_FRA_doc_url.csv')
    df_FRA_doc_url['date'] = pd.to_datetime(df_FRA_doc_url['date'])
    start_date = df_FRA_doc_url['date'].max().strftime('%Y-%m-%d')
else:
    df_FRA_doc_url = pd.DataFrame(columns=['series', 'subseries', 'date', 'url', 'page'])
    start_date = START_DATE
    df_FRA_doc_url.to_csv(DATA_FOLDER + 'df_FRA_doc_url.csv', index=False)

for key, value in dict_series_id.items():
    series = key
    id_series = value['id']
    dict_subseries_id = value['subseries']
    for key, value in dict_subseries_id.items():
        subseries = key
        id_subseries = value['id']
        if df_FRA_doc_url[df_FRA_doc_url['subseries'] == subseries]['page'].min() == 1:
            continue
        not_first_page = True
        while not_first_page:
            print(series, '|', subseries)
            try:
                df_FRA_doc_url = pd.read_csv(DATA_FOLDER + 'df_FRA_doc_url.csv')
                df_FRA_doc_url['date'] = pd.to_datetime(df_FRA_doc_url['date'])
                start_date = df_FRA_doc_url[(df_FRA_doc_url['series'] == series) & (df_FRA_doc_url['subseries'] == subseries)]['date'].max()
                if pd.isna(start_date):
                    start_date = START_DATE
                else:
                    start_date = start_date.strftime('%Y-%m-%d')
                
                driver = webdriver.Chrome(options=options, service=service)
                driver.get(f'https://railroads.dot.gov/elibrary-search?field_effective_date%5Bmin%5D={start_date}&sort_by=field_effective_date&items_per_page=50&f%5B0%5D=document_series%3A{id_subseries}')
                time.sleep(TIME_WAIT)
                
                # go to last page (the oldest doc)
                _ = last_button_clicked(driver)
                time.sleep(TIME_WAIT)
                
                while not_first_page:
                    _ = driver.save_screenshot('web_screenshot.png')
                    
                    list_date = []
                    list_href = []
                    list_row = find_docs(driver)
                    list_row = list_row[::-1]
                    for row in list_row:
                        date = row.find_element(By.XPATH, './/div[@class="col-xs-12 col-md-1 order-first col-date d-flex flex-md-column align-items-center py-2"]')
                        date = date.text.replace('\n', ' ')
                        date = pd.to_datetime(date)
                        list_date.append(date)
                        title = row.find_element(By.XPATH, './/h2[@class="title elibrary--title"]/a')
                        href = title.get_attribute('href')
                        list_href.append(href)
                    page_num = get_page_num(driver)
                    df_FRA_doc_url_temp = pd.DataFrame({'series': series, 'subseries': subseries, 'date': list_date, 'url': list_href, 'page': page_num})
                    df_FRA_doc_url = pd.concat([df_FRA_doc_url, df_FRA_doc_url_temp])
                    df_FRA_doc_url = df_FRA_doc_url.drop_duplicates().reset_index(drop=True)
                    df_FRA_doc_url.to_csv(DATA_FOLDER + 'df_FRA_doc_url.csv', index=False)
                    
                    if page_num == 1:
                        not_first_page = False
                    else:
                        _ = prev_button_clicked(driver)
                        time.sleep(TIME_WAIT)
            except Exception as e:
                print(e)
                pass
        driver.close()
