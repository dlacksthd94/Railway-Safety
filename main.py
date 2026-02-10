import os
from modules import (
    build_config, scrape_news, filter_news, convert_to_json, extract_keywords, 
    merge_record_retrieval, scrape_image, scrape_image_seq, scrape_3D
)
from modules.metrics import get_acc_table, get_cov_table, get_stats

print('###########################################################################')
print('###########################################################################')

############### config
cfg = build_config()
print(cfg.path.dir_conversion)
print(cfg.path.dir_retrieval)
print('------------Configuration DONE!!------------')

# ############### news scraping
# df_record_news, df_news_articles = scrape_news(cfg)
# print('------------Scraping DONE!!------------')


# # ############### news filtering
# df_news_articles_filter = filter_news(cfg)
# print('------------Filtering DONE!!------------')


# ############### convert to json
# dict_form57, dict_form57_group = convert_to_json(cfg)
# print('------------Conversion DONE!!------------')


# ############### extract keywords
# df_retrieval = extract_keywords(cfg)
# print('------------Retrieval DONE!!------------')


############### match samples manually by running following command in terminal (ONLY ONE-TIME TASK FOR EVALUATION)
# streamlit run match_record_news.py
assert os.path.exists(cfg.path.df_match)
print('------------Matching DONE!!------------')


############### annotate samples manually by running following command in terminal (ONLY ONE-TIME TASK FOR EVALUATION)
# streamlit run annotate_news.py
assert os.path.exists(cfg.path.df_annotate)
print('------------Annotating DONE!!------------')


############### merge news-record pair and retrieval results
df_record_retrieval = merge_record_retrieval(cfg)
print('------------Merging DONE!!------------')


############### calculate the accuracy and coverage
assert os.path.exists(cfg.path.dict_idx_mapping), "Must map index names shared accross the models with form transcription manually"

# list_answer_type_selected = ['digit', 'text', 'choice']
# df_acc, acc = get_acc_table(list_answer_type_selected, cfg)
# print(f'ACCURACY\n{" + ".join(list_answer_type_selected)}:\t', acc)

# list_answer_type_selected = ['choice']
# df_acc, acc = get_acc_table(list_answer_type_selected, cfg)
# print(f'ACCURACY\n{" + ".join(list_answer_type_selected)}:\t', acc)

# list_answer_type_selected = ['digit']
# df_acc, acc = get_acc_table(list_answer_type_selected, cfg)
# print(f'ACCURACY\n{" + ".join(list_answer_type_selected)}:\t', acc)

# list_answer_type_selected = ['text']
# df_acc, acc = get_acc_table(list_answer_type_selected, cfg)
# print(f'ACCURACY\n{" + ".join(list_answer_type_selected)}:\t', acc)

# list_answer_type_selected = ['digit', 'text', 'choice']
# df_cov, cov = get_cov_table(list_answer_type_selected, cfg)
# print(f'COVERAGE\n{" + ".join(list_answer_type_selected)}:\t', cov)

# get_stats(df_acc, cfg)

print('------------Metrics DONE!!------------')


# ############### scrape crossing images from mapillary (ONLY ONE-TIME TASK)
# df_image = scrape_image(cfg)
# df_image_seq = scrape_image_seq(cfg)
# print('------------Scraping Images DONE!!------------')


############### scrape 3D reconstruction from mapillary (ONLY ONE-TIME TASK)
df_3D = scrape_3D(cfg)


############### 

# # ############### merge retrieval-record
# df_rci = merge_news_image(cfg)

print('###########################################################################')
print('###########################################################################')
