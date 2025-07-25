import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser(
    description="Run main.py"
)

# Positional arguments (order matters)
# parser.add_argument("file", help="Path to input file")
# parser.add_argument("count", type=int, help="Number of iterations")

# Optional (flag-style) arguments
parser.add_argument(
    "--c_api",
    # choices=["fast", "safe"],
    # default="safe",
    # help="Processing mode (default: safe)"
)

parser.add_argument(
    "--c_model",
    type=str,
    # default=0.5,
    # help="Confidence threshold (default: 0.5)"
)

parser.add_argument(
    "--c_n_generate",
    type=int
    # action="store_true",
    # help="Enable verbose output"
)

parser.add_argument("--c_json_source")

# parser.add_argument("--r_question_batch")

args = parser.parse_args()


############ config
from config import Config, ConversionConfig, RetrievalConfig

### ConversionConfig
# config_conversion = ConversionConfig(api=None, model=None, n_generate=None, json_source='csv')
# config_conversion = ConversionConfig(api='OpenAI', model='o4-mini', n_generate=8, json_source='pdf')
config_conversion = ConversionConfig(api='Huggingface', model='Qwen/Qwen2.5-VL-7B-Instruct', n_generate=4, json_source='img')
# config_conversion = ConversionConfig(api='Huggingface', model='OpenGVLab/InternVL3-8B-hf', n_generate=4, json_source='img')
# config_conversion = ConversionConfig(api=args.c_api, model=args.c_model, n_generate=args.c_n_generate, json_source=args.c_json_source)

### RetrievalConfig
config_retrieval = RetrievalConfig(api='Huggingface', model='microsoft/Phi-4-mini-instruct', n_generate=1, question_batch='single')
# config_retrieval = RetrievalConfig(api='Huggingface', model='microsoft/Phi-4-mini-instruct', n_generate=1, question_batch='group')
config = Config(conversion=config_conversion, retrieval=config_retrieval)

print('------------Configuration DONE!!------------')

############ paths
from utils import make_dir

DIR_DATA_ROOT = 'data'
DIR_DATA_JSON = 'json'
DIR_DATA_RESULT = 'result'

FN_DF_NEWS_LABEL = 'df_news_label.csv'

NM_FORM57 = 'FRA F 6180.57 (Form 57) form only'
FN_FORM57_PDF = f'{NM_FORM57}.pdf'
FN_FORM57_IMG = f'{NM_FORM57}.jpg'
FN_FORM57_CSV = '250424 Highway-Rail Grade Crossing Incident Data (Form 57).csv'

FN_FORM57_JSON = f'form57.json'
FN_FORM57_JSON_GROUP = f'form57_group.json'

FN_DF_FORM57_RETRIEVAL = f'df_form57_retrieval.csv'

FN_DF_MATCH = 'df_match.csv'

conversion_model_replaced = config.conversion.model.replace('/', '@')
retrieval_model_replaced = config.retrieval.model.replace('/', '@')
dir_config_json = f'{config.conversion.json_source}_{config.conversion.api}_{conversion_model_replaced}_{config.conversion.n_generate}'
path_dir_config_json = os.path.join(DIR_DATA_ROOT, DIR_DATA_JSON, dir_config_json)
make_dir(path_dir_config_json)
dir_config_result = f'{config.retrieval.api}_{retrieval_model_replaced}_{config.retrieval.n_generate}_{config.retrieval.question_batch}'
path_dir_config_result = os.path.join(path_dir_config_json, DIR_DATA_RESULT, dir_config_result)
make_dir(path_dir_config_result)

path_df_news_label = os.path.join(DIR_DATA_ROOT, FN_DF_NEWS_LABEL)
path_form57_csv = os.path.join(DIR_DATA_ROOT, FN_FORM57_CSV)
path_form57_pdf = os.path.join(DIR_DATA_ROOT, FN_FORM57_PDF)
path_form57_img = os.path.join(DIR_DATA_ROOT, FN_FORM57_IMG)
path_form57_json = os.path.join(path_dir_config_json, FN_FORM57_JSON)
path_form57_json_group = os.path.join(path_dir_config_json, FN_FORM57_JSON_GROUP)
path_df_form57_retrieval = os.path.join(path_dir_config_result, FN_DF_FORM57_RETRIEVAL)
path_df_match = os.path.join(DIR_DATA_ROOT, 'df_match.csv')

print('------------Setting path DONE!!------------')

############ convert to json
from convert_report_to_json import csv_to_json, pdf_to_json, img_to_json

if config.conversion.json_source == 'csv':
    dict_form57 = csv_to_json(path_form57_csv, path_form57_json)
elif config.conversion.json_source == 'pdf':
    dict_form57, dict_form57_group = pdf_to_json(path_form57_pdf, path_form57_json, path_form57_json_group, config.conversion)
elif config.conversion.json_source == 'img':
    dict_form57, dict_form57_group = img_to_json(path_form57_img, path_form57_json, path_form57_json_group, config.conversion)

print('------------Conversion DONE!!------------')

############ extract keywords
from extract_keywords import extract_keywords

df_retrieval = extract_keywords(path_form57_json, path_form57_json_group, path_df_form57_retrieval, path_df_news_label, config.retrieval)

print('------------Retrieval DONE!!------------')

############ match news and csv file using retrieved information
assert os.path.exists(path_df_match)
df_match = pd.read_csv(path_df_match)
df_match = df_match[df_match['match'] == 1]

print('------------Matching DONE!!------------')

############ calculate the accuracy
