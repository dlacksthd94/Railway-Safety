import os
import shutil
import zipfile
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

######################### find Q&A files ###########################
if not os.path.exists(BIGDATA_FOLDER + 'QA/'):
    os.mkdir(BIGDATA_FOLDER + 'QA/')

bool_qa = df_FRA_doc['file_name'].fillna('').str.match(r'.*((?<!CE)QA|Q&A|QnA|QNA|FAQ|question|Qs and As|qa|q&a|qna|faq|qs and as|qsas).*')
for i, row in df_FRA_doc[bool_qa].iterrows():
    series, subseries, file_name = row[['series', 'subseries', 'file_name']]
    qa_file_path_from = BIGDATA_FOLDER + DOC_FRA_FOLDER + series + '/' + subseries + '/' + file_name
    if os.path.exists(qa_file_path_from):
        1
        qa_file_path_to = BIGDATA_FOLDER + 'QA/' + file_name
        shutil.copyfile(qa_file_path_from, qa_file_path_to)

# Create a zip file
qa_zip_name = BIGDATA_FOLDER + 'QA'
qa_folder_path = BIGDATA_FOLDER + 'QA/'
shutil.make_archive(qa_zip_name, 'zip', qa_folder_path)