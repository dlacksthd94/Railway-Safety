import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from scipy.spatial.distance import cdist
import pathlib
import shutil
from .utils import (prepare_df_record, prepare_df_crossing,
                    prepare_df_retrieval, prepare_df_match, prepare_df_image_seq, prepare_df_image)

def merge_record_retrieval(cfg):
    df_retrieval = pd.read_csv(cfg.path.df_retrieval)
    df_match = prepare_df_match(cfg)
    
    df_retrieval_drop = df_retrieval.set_index('news_id')
    idx_content_retrieval = df_retrieval_drop.columns.get_loc('content')
    df_retrieval_drop = df_retrieval_drop.iloc[:, idx_content_retrieval + 1:] # type: ignore
    df_record_retrieval = df_match.merge(df_retrieval_drop, left_on='news_id', right_index=True, how='inner')
    df_record_retrieval.to_csv(cfg.path.df_record_retrieval, index=False)
    return df_record_retrieval

def merge_news_image(cfg):
    df_record = prepare_df_record(cfg)
    df_record = df_record[['Report Key', 'Grade Crossing ID']].set_index('Report Key')
    
    df_match = prepare_df_match(cfg)
    df_match = df_match[['news_id', 'report_key']]
    df_retrieval = prepare_df_retrieval(cfg)

    df_image_seq = prepare_df_image_seq(cfg)
    crossings_w_image = df_image_seq['crossing_id'].unique()
    df_image = prepare_df_image(cfg)

    df_news_image = df_match.merge(df_record, left_on='report_key', right_index=True)
    df_news_image = df_news_image.rename({'Grade Crossing ID': 'crossing_id'}, axis=1)
    df_news_image = df_news_image[df_news_image['crossing_id'].isin(crossings_w_image)]
    print(f"https://news.google.com/read/{df_news_image['news_id'].values[0]}")
    # df_news_image.merge(df_image_seq, on='crossing_id')
    
    df_image_seq.merge(df_record, left_on='report_key')

    
    
    # df_record_retrieval = pd.read_csv(cfg.path.df_record_retrieval)
    # idx_content = df_record_retrieval.columns.get_loc('content')
    # df_record_retrieval_drop = df_record_retrieval.iloc[:, :idx_content + 1] # type: ignore
    
    df_crossing = prepare_df_crossing(cfg).set_index('CROSSING')

    df_msls_meta, df_dir_info = _concat_msls(cfg)

    dists = cdist(df_msls_meta[['lon', 'lat']], df_crossing[['LONGITUD', 'LATITUDE']], metric='euclidean')
    df_dist = pd.DataFrame(dists, index=df_msls_meta['key'], columns=df_crossing.index)
    df_min_dist = pd.DataFrame(index=df_dist.columns)
    df_min_dist['key'] = df_dist.idxmin(axis=0)
    df_min_dist['dist'] = df_dist.min(axis=0)
    df_min_dist = df_min_dist.sort_values('dist')
    
    df_min_dist = df_min_dist.merge(df_msls_meta[['tt', 'city', 'dq', 'key']], on='key')

    df_min_dist = df_min_dist[(0.0001 < df_min_dist['dist']) & (df_min_dist['dist'] <= 0.0002)]
    for i, row in df_min_dist.iterrows():
        key, dist, tt, city, dq = row
        vol = df_dir_info[(df_dir_info['tt'] == tt) & (df_dir_info['city'] == city) & (df_dir_info['dq'] == dq)]['vol'].squeeze()
        img_path = os.path.join(cfg.path.dir_msls, vol, tt, city, dq, 'images', key + '.jpg')
        dst_path = os.path.join(cfg.path.dir_msls_crossing, key + '.jpg')
        shutil.copy(img_path, dst_path)
    
    return df_min_dist

def _concat_msls(cfg):
    us_cities = cfg.crss.us_cities
    
    ############### info of image directories
    df_dir_info_cols = ['vol', 'tt', 'city', 'dq', 'dir']
    df_dir_info = pd.DataFrame(columns=df_dir_info_cols)
    for img_dir in cfg.path.dirs_msls_image:
        img_de = pathlib.Path(img_dir)
        for tt_de in os.scandir(img_dir):
            for city_de in os.scandir(tt_de):
                if city_de.name in us_cities:
                    for dq_de in os.scandir(city_de):
                        assert dq_de.name in ['database', 'query']
                        df_dir_info.loc[len(df_dir_info)] = [img_de.name, tt_de.name, city_de.name, dq_de.name, dq_de.path]
    df_dir_info = df_dir_info.reset_index(names='dir_id')
    
    # df_img_cols = ['dir_id', 'img']
    # df_img = pd.DataFrame(columns=df_img_cols)
    # for i, row in df_dir_info.iterrows():
    #     dir_id = row.pop('dir_id')
    #     dir_path = row.pop('dir')
    #     dq_dir = os.path.join(cfg.path.dir_msls, '/'.join(row.values))
    #     for img_de in os.scandir(dq_dir):
    #         assert img_de.name == 'images'
    #         for img in os.scandir(img_de.path):
    #             assert img.name.endswith('jpg')
    #         imgs = os.listdir(img_de)
    #         df_img_temp = pd.DataFrame(list(zip([dir_id] * len(imgs), imgs)), columns=df_img_cols)
    #         df_img = pd.concat([df_img, df_img_temp], axis=0, ignore_index=True)
    # df_img['img'] = df_img['img'].str.split('.').str[0]
    
    ############### concat meta
    df_meta_cols = ['key', 'lon', 'lat', 'ca', 'captured_at', 'pano', 'tt', 'city', 'dq']
    df_meta = pd.DataFrame(columns=df_meta_cols)
    for tt_de in os.scandir(cfg.path.dir_msls_meta):
        for city_de in os.scandir(tt_de):
            if city_de.name in us_cities:
                for dq_de in os.scandir(city_de):
                    if 'raw.csv' in os.listdir(dq_de):
                        df_meta_temp = pd.read_csv(os.path.join(dq_de.path, 'raw.csv'), index_col=0)
                        df_meta_temp['tt'] = tt_de.name
                        df_meta_temp['city'] = city_de.name
                        df_meta_temp['dq'] = dq_de.name
                        df_meta = pd.concat([df_meta, df_meta_temp], ignore_index=True)
    df_meta['city'].unique()
    df_meta.to_csv(cfg.path.df_msls_meta)

    return df_meta, df_dir_info