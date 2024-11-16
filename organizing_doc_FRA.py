import os
import shutil
import pandas as pd

DATA_FOLDER = 'data/'
BIGDATA_FOLDER = '/data2/clim090/RailwaySafety/'
DOC_FRA_FOLDER = 'doc_FRA/'

df_FRA_doc = pd.read_csv(DATA_FOLDER + 'df_FRA_doc.csv')
df_FRA_doc['date'] = pd.to_datetime(df_FRA_doc['date'])

for file_name in os.listdir(BIGDATA_FOLDER + DOC_FRA_FOLDER):
    if not file_name.endswith('pdf'):
        continue
    file_name_from = BIGDATA_FOLDER + DOC_FRA_FOLDER + file_name
    file_name_to = BIGDATA_FOLDER + DOC_FRA_FOLDER + file_name.replace('  ', ' ')
    os.rename(file_name_from, file_name_to)

for i, (series, subseries, file_name) in df_FRA_doc[['series', 'subseries', 'file_name']].iterrows():
    series_folder = BIGDATA_FOLDER + DOC_FRA_FOLDER + series
    if not os.path.exists(series_folder):
        os.mkdir(series_folder)
    subseries_folder = BIGDATA_FOLDER + DOC_FRA_FOLDER + series + '/' + subseries
    if not os.path.exists(subseries_folder):
        os.mkdir(subseries_folder)
    if pd.isna(file_name):
        continue
    file_path_from = BIGDATA_FOLDER + DOC_FRA_FOLDER + file_name
    file_path_to = BIGDATA_FOLDER + DOC_FRA_FOLDER + series + '/' + subseries + '/' + file_name
    if os.path.exists(file_path_from):
        os.rename(file_path_from, file_path_to)
