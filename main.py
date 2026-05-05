import os
from modules import (
    build_config, scrape_news, filter_news, convert_to_json, extract_keywords, merge_record_retrieval,
    scrape_image, scrape_image_seq, preprocess_image,
    scrape_3D
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


# ############### match samples manually by running following command in terminal (ONLY ONE-TIME TASK FOR EVALUATION)
# # streamlit run match_record_news.py
# assert os.path.exists(cfg.path.df_match)
# print('------------Matching DONE!!------------')


# ############### annotate samples manually by running following command in terminal (ONLY ONE-TIME TASK FOR EVALUATION)
# # streamlit run annotate_news.py
# assert os.path.exists(cfg.path.df_annotate)
# print('------------Annotating DONE!!------------')


# ############### merge news-record pair and retrieval results
# df_record_retrieval = merge_record_retrieval(cfg)
# print('------------Merging DONE!!------------')


# ############### calculate the accuracy and coverage
# assert os.path.exists(cfg.path.dict_idx_mapping), "Must map index names shared accross the models with form transcription manually"

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

# print('------------Metrics DONE!!------------')


# ############### scrape crossing images from mapillary (ONLY ONE-TIME TASK)
# df_image = scrape_image(cfg)
# df_image_seq = scrape_image_seq(cfg)
# print('------------Scraping Images DONE!!------------')


############### preprocess images (ONLY ONE-TIME TASK)
# yoloe-11l-seg, yoloe-v8l-seg, yoloe-26x-seg, IDEA-Research/grounding-dino-tiny, IDEA-Research/grounding-dino-base
# preprocess_image(cfg, model_name='yoloe-26x-seg', confidence_threshold=0.5, prompt_type='text', text_class=['traffic sign', 'traffic light'])

# preprocess_image(cfg, model_name='yoloe-26x-seg', confidence_threshold=0.1, prompt_type='text', text_class=["X-shaped white traffic sign with black text", "Two white rectangular boards crossed in an X-shape", "X-shaped railroad crossing sign on a metal pole", "White wooden or metal planks forming a cross with 'RAILROAD CROSSING' text", "X-shaped sign with small red reflectors on the edges"]) # not bad, but not good enough
# preprocess_image(cfg, model_name='yoloe-26x-seg', confidence_threshold=0.1, prompt_type='text', text_class=["Long horizontal pole with red and white diagonal stripes", "Long slender barrier arm positioned across the road", "Retractable safety gate arm at a railroad crossing", "Gate arm with small red warning lights attached", "Reflective striped wooden or fiberglass barrier pole", "Lowered railroad crossing gate blocking the lane"]) # very bad

# preprocess_image(cfg, model_name='yoloe-26x-seg', confidence_threshold=0.1, prompt_type='visual', visual_class='crossbuck_4.jpg')

# preprocess_image(cfg, model_name='IDEA-Research/grounding-dino-base', confidence_threshold=0.3, prompt_type='text', text_class=["traffic sign", "traffic light"]) # grounding dino preprocesses text inputs as '. '.join(list_of_labels), not treating each label as an individual token.
# preprocess_image(cfg, model_name='IDEA-Research/grounding-dino-base', confidence_threshold=0.3, prompt_type='text', text_class=["gate arm", "barrier arm", "lifted gate arm", "lifted barrier arm"])

preprocess_image(cfg, model_name='facebook/sam-vit-huge', confidence_threshold=0.5, prompt_type='visual', visual_class='crossbuck_4.jpg')
# preprocess_image(cfg, model_name='facebook/sam-vit-huge', confidence_threshold=0.5, prompt_type='visual', visual_class=['crossbuck_1.jpg', 'crossbuck_2.jpg', 'crossbuck_3.jpg', 'crossbuck_4.jpg'])

print('------------Preprocessing Images DONE!!------------')

# ############### scrape 3D reconstruction from mapillary (ONLY ONE-TIME TASK)
# df_3D = scrape_3D(cfg)


############### 

# # ############### merge retrieval-record
# df_rci = merge_news_image(cfg)

print('###########################################################################')
print('###########################################################################')