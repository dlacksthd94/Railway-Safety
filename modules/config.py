from dataclasses import dataclass, asdict, astuple
from typing import Final
import os
import argparse
import re
import json
from .utils import make_dir, remove_dir, sanitize_model_path

# directories
DN_DATA_ROOT: Final[str] = "data"
DN_DATA_JSON: Final[str] = "json"
DN_DATA_RESULT: Final[str] = "result"
DN_MAPILLARY: Final[str] = 'mapillary'
DN_MSLS: Final[str] = 'street_level_seq'
DNS_MSLS_IMAGE: Final[tuple[str, ...]] = tuple(f'msls_images_vol_{i}' for i in range(1, 7))
DN_MSLS_META: Final[str] = 'msls_metadata'
DN_MSLS_CROSSING: Final[str] = 'msls_crossing'
DN_SCRAPED_IMAGES: Final[str] = 'scraped_image'
DN_IMAGE_SEQ: Final[str] = 'image_seq'

# files
FN_DICT_API_KEY: Final[str] = 'dict_api_key.json'

FN_DF_RECORD: Final[str] = "250821 Highway-Rail Grade Crossing Incident Data (Form 57).csv"

FN_DF_RECORD_NEWS: Final[str] = "df_record_news.csv"
FN_DF_NEWS_ARTICLES: Final[str] = "df_news_articles.csv"

FN_DF_NEWS_ARTICLES_SCORE: Final[str] = "df_news_articles_score.csv"
FN_DF_NEWS_ARTICLES_FILTER: Final[str] = "df_news_articles_filter.csv"

NM_FORM57: Final[str] = "FRA F 6180.57 (Form 57) form only"
FN_FORM57_PDF: Final[str] = f"{NM_FORM57}.pdf"
FN_FORM57_IMG: Final[str] = f"{NM_FORM57}.jpg"

FN_FORM57_JSON: Final[str] = "form57.json"
FN_FORM57_JSON_GROUP: Final[str] = "form57_group.json"
FN_FORM57_ANNOTATED: Final[str] = 'annotated_form.png'

FN_DICT_ANSWER_PLACES: Final[str] = "dict_answer_places.jsonc"
FN_DICT_IDX_MAPPING: Final[str] = "dict_idx_mapping.jsonc"
FN_DICT_COL_INDEXING: Final[str] = "dict_col_indexing.jsonc"

FN_DF_RETRIEVAL: Final[str] = "df_retrieval.csv"
FN_DF_MATCH: Final[str] = "df_match.csv"
FN_DF_ANNOTATE: Final[str] = "df_annotate.csv"
FN_DF_RECORD_RETRIEVAL: Final[str] = 'df_record_retrieval.csv'

FN_DF_CROSSING: Final[str] = '251009 NTAD_Railroad_Grade_Crossings_1739202960140128164.csv'
FN_DF_MSLS_META: Final[str] = 'df_msls_meta.csv'
FN_DF_IMAGE: Final[str] = 'df_image.csv'
FN_DF_IMAGE_SEQ: Final[str] = 'df_image_seq.csv'

# configurations
NEWS_CRAWLERS: Final[tuple[str, ...]] = ("np_url", "tf_url", "rd_url", "gs_url", "np_html", "tf_html", "rd_html", "gs_html")
TARGET_STATES: Final[tuple[str, ...]] = ('California',)
START_DATE: Final[str] = '2000-01-01'

CONVERSION_API_MODEL_CHOICES: Final[dict[str, list[str]]] = {
    'Google_DocAI': ['form_parser', 'layout_parser'],
    'AWS_Textract': ['textract'],
    'Azure_FormRecognizer': ['form_recognizer'],
    "OpenAI": ["o4-mini"],
    "Huggingface": ["Qwen/Qwen2.5-VL-72B-Instruct",
                    "Qwen/Qwen3-VL-32B-Instruct",
                    "OpenGVLab/InternVL3_5-38B-HF",
                    'OpenGVLab/InternVL3-78B-hf',
                    ],
    "None": ["None"],
}
CONVERSION_JSON_SOURCE_CHOICES: Final[list[str]] = ["csv", "pdf", "img", "None"]
RETRIEVAL_API_MODELS_CHOICES: Final[dict[str, list[str]]] = {
    "Huggingface": ["microsoft/phi-4", "Qwen/Qwen2.5-VL-72B-Instruct"],
    "OpenAI": ["o4-mini"],
}
RETRIEVAL_BATCH_CHOICES: Final[list[str]] = ["single", "group", "all"]
N_GENERATE_RANGE: Final[range] = range(9)

US_CITIES: Final[tuple[str, ...]] = ('miami', 'austin', 'boston', 'phoenix', 'sf') # miami images don't provide geographic info (lon, lat)
IMG_SEARCH_FIELDS: Final[tuple[str, ...]] = ("id",)
IMG_DETAIL_FIELDS: Final[tuple[str, ...]] = (
    "id",
    "altitude", "computed_altitude", "geometry", "computed_geometry",  # geographic info
    "captured_at",  # timestamp
    "height", "width", "thumb_original_url",  # image
    "compass_angle", "computed_compass_angle", "computed_rotation", "exif_orientation", # orientation info
    "camera_type", "is_pano", "make", "model",  # camera info
    "sequence",  # sequence info
    "atomic_scale", "merge_cc", "mesh", "sfm_cluster", # SfM info
    "creator",  # uploader info
    # "detections", # object detection - can be fetched using other API: https://www.mapillary.com/developer/api-documentation#detection
)
BBOX_OFFSET: Final[float] = 0.0002 # 0.00001 ≒ 1.11 meters
N_IMG: Final[int] = int((BBOX_OFFSET * 100000) ** 2 * 3) # 3 images per 1m²

def parse_args() -> argparse.Namespace:
    """Create an argument parser for building configs from the CLI and parse CLI args."""
    parser = argparse.ArgumentParser(description="Project configuration parser")

    g_conv = parser.add_argument_group("conversion")
    g_conv.add_argument(
        "--c_api",
        type=str,
        choices=list(CONVERSION_API_MODEL_CHOICES.keys()),
        required=True,
        help="API to use for form transcription"
    )
    g_conv.add_argument(
        "--c_model",
        type=str,
        choices=sorted([m for ms in CONVERSION_API_MODEL_CHOICES.values() for m in ms]),
        required=True,
        help="Model to use for form transcription"
    )
    g_conv.add_argument(
        "--c_n_generate",
        type=int,
        choices=N_GENERATE_RANGE,
        required=True,
        help="Number of transcription sample generations before aggregation"
    )
    g_conv.add_argument(
        "--c_json_source",
        type=str,
        choices=CONVERSION_JSON_SOURCE_CHOICES,
        required=True,
        help="Source of JSON data"
    )
    g_conv.add_argument(
        "--c_seed",
        type=int,
        required=True,
        help="Simulation seed number for reproducibility"
    )

    g_ret = parser.add_argument_group("retrieval")
    g_ret.add_argument(
        "--r_api",
        type=str,
        choices=list(RETRIEVAL_API_MODELS_CHOICES.keys()),
        required=True,
        help="API to use for information retrieval"
    )
    g_ret.add_argument(
        "--r_model",
        type=str,
        choices=sorted([m for ms in RETRIEVAL_API_MODELS_CHOICES.values() for m in ms]),
        required=True,
        help="Model to use for information retrieval",
    )
    g_ret.add_argument(
        "--r_n_generate",
        type=int,
        choices=N_GENERATE_RANGE,
        required=False,
        default=1,
        help="Number of QAs before aggregation"
    )
    g_ret.add_argument(
        "--r_question_batch",
        type=str,
        choices=RETRIEVAL_BATCH_CHOICES,
        required=True,
        help="Batching strategy for questions"
    )

    args = parser.parse_args()

    # sanity check
    if args.c_model not in CONVERSION_API_MODEL_CHOICES[args.c_api]:
        parser.error(f"Model '{args.c_model}' is invalid for API '{args.c_api}'. "
                     f"Allowed models: {CONVERSION_API_MODEL_CHOICES[args.c_api]}")
    
    if args.r_model not in RETRIEVAL_API_MODELS_CHOICES[args.r_api]:
        parser.error(f"Model '{args.r_model}' is invalid for API '{args.r_api}'. "
                     f"Allowed models: {RETRIEVAL_API_MODELS_CHOICES[args.r_api]}")
    
    if args.c_json_source == 'csv':
        assert args.c_api == 'None' and args.c_model == 'None' and args.c_n_generate == 0 and args.r_question_batch == 'single'
    
    return args

@dataclass(frozen=True)
class BaseConfig:
    api: str
    model: str
    n_generate: int
    
    def to_dict(self):
        return asdict(self)

    def to_tuple(self):
        return astuple(self)

@dataclass(frozen=True)
class ScrapingConfig:
    target_states: tuple[str, ...]
    start_date: str
    news_crawlers: tuple[str, ...]

    n_img: int
    img_search_fields: tuple[str, ...]
    img_detail_fields: tuple[str, ...]
    bbox_offset: float

    def __post_init__(self):
        """sanity check"""
        for state in self.target_states:
            assert state in ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'District Of Columbia', 
                                     'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 
                                     'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 
                                     'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 
                                     'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 
                                     'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']
        assert bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.start_date))

@dataclass(frozen=True)
class TableConfig:
    path: str
    columns: list[str]

@dataclass(frozen=True)
class ConversionConfig(BaseConfig):
    json_source: str
    seed: int

@dataclass(frozen=True)
class RetrievalConfig(BaseConfig):
    question_batch: str

@dataclass(frozen=True)
class PathConfig:
    # generated directories
    dir_conversion: str
    dir_retrieval: str
    dir_msls: str
    dirs_msls_image: tuple[str, ...]
    dir_msls_meta: str
    dir_msls_crossing: str
    dir_scraped_images: str
    dir_image_seq: str
    
    # files
    dict_api_key: str

    df_record: str
    df_record_news: str
    df_news_articles: str

    df_news_articles_score: str
    df_news_articles_filter: str
    
    form57_pdf: str
    form57_img: str
    
    form57_json: str
    form57_json_group: str
    form57_annotated: str
    
    df_retrieval: str
    df_match: str
    df_annotate: str
    df_record_retrieval: str
    
    dict_answer_places: str
    dict_idx_mapping: str
    dict_col_indexing: str
    
    df_crossing: str
    df_msls_meta: str
    df_image: str
    df_image_seq: str

@dataclass()
class APIkeyConfig:
    openai: str
    textract: str
    mapillary: str

@dataclass()
class CrossingConfig:
    us_cities: tuple[str, ...]

@dataclass(frozen=True)
class Config:
    scrp: ScrapingConfig
    conv: ConversionConfig
    retr: RetrievalConfig
    crss: CrossingConfig
    path: PathConfig
    apikey: APIkeyConfig

def _compute_paths(conv_cfg: ConversionConfig, retr_cfg: RetrievalConfig) -> PathConfig:
    dn_conversion = f"{conv_cfg.json_source}_{conv_cfg.api}_{conv_cfg.model}_{conv_cfg.n_generate}"
    dp_conversion = os.path.join(DN_DATA_ROOT, DN_DATA_JSON, dn_conversion, str(conv_cfg.seed))
    make_dir(dp_conversion)

    dn_retrieval = f"{retr_cfg.question_batch}_{retr_cfg.api}_{retr_cfg.model}_{retr_cfg.n_generate}"
    dp_retrieval = os.path.join(dp_conversion, DN_DATA_RESULT, dn_retrieval)
    make_dir(dp_retrieval)

    dp_mapillary = os.path.join(DN_DATA_ROOT, DN_MAPILLARY)
    dp_msls = os.path.join(dp_mapillary, DN_MSLS)
    dp_msls_meta = os.path.join(dp_msls, DN_MSLS_META)
    dp_msls_crossing = os.path.join(dp_msls, DN_MSLS_CROSSING)
    remove_dir(dp_msls_crossing)
    make_dir(dp_msls_crossing)

    dp_scraped_images = os.path.join(dp_mapillary, DN_SCRAPED_IMAGES)
    make_dir(dp_scraped_images)

    dp_image_seq = os.path.join(dp_mapillary, DN_IMAGE_SEQ)
    make_dir(dp_image_seq)

    return PathConfig(
        dir_conversion=dp_conversion,
        dir_retrieval=dp_retrieval,
        dir_msls=dp_msls,
        dirs_msls_image=tuple(map(lambda dn: os.path.join(dp_msls, dn), DNS_MSLS_IMAGE)),
        dir_msls_meta=dp_msls_meta,
        dir_msls_crossing=dp_msls_crossing,
        dir_scraped_images=dp_scraped_images,
        dir_image_seq=dp_image_seq,

        dict_api_key=os.path.join(DN_DATA_ROOT, FN_DICT_API_KEY),

        df_record=os.path.join(DN_DATA_ROOT, FN_DF_RECORD),
        df_record_news=os.path.join(DN_DATA_ROOT, FN_DF_RECORD_NEWS),
        df_news_articles=os.path.join(DN_DATA_ROOT, FN_DF_NEWS_ARTICLES),
        
        df_news_articles_score=os.path.join(DN_DATA_ROOT, FN_DF_NEWS_ARTICLES_SCORE),
        df_news_articles_filter=os.path.join(DN_DATA_ROOT, FN_DF_NEWS_ARTICLES_FILTER),
        
        form57_pdf=os.path.join(DN_DATA_ROOT, FN_FORM57_PDF),
        form57_img=os.path.join(DN_DATA_ROOT, FN_FORM57_IMG),

        form57_json=os.path.join(dp_conversion, FN_FORM57_JSON),
        form57_json_group=os.path.join(dp_conversion, FN_FORM57_JSON_GROUP),
        form57_annotated=os.path.join(dp_conversion, FN_FORM57_ANNOTATED),
        
        dict_answer_places=os.path.join(dp_conversion, FN_DICT_ANSWER_PLACES),
        dict_idx_mapping=os.path.join(dp_conversion, FN_DICT_IDX_MAPPING),
        dict_col_indexing=os.path.join(DN_DATA_ROOT, FN_DICT_COL_INDEXING),
        
        df_retrieval=os.path.join(dp_retrieval, FN_DF_RETRIEVAL),
        df_match=os.path.join(DN_DATA_ROOT, FN_DF_MATCH),
        df_annotate=os.path.join(DN_DATA_ROOT, FN_DF_ANNOTATE),
        df_record_retrieval=os.path.join(dp_retrieval, FN_DF_RECORD_RETRIEVAL),
        
        df_crossing=os.path.join(DN_DATA_ROOT, FN_DF_CROSSING),
        df_msls_meta=os.path.join(dp_msls, FN_DF_MSLS_META),
        df_image=os.path.join(dp_scraped_images, FN_DF_IMAGE),
        df_image_seq=os.path.join(dp_image_seq, FN_DF_IMAGE_SEQ),
    )

def _load_api_key(path_cfg: PathConfig) -> APIkeyConfig:
    with open(path_cfg.dict_api_key, 'r') as f:
        dict_api_key = json.load(f)
    dict_api_key = {api: info['key'] for api, info in dict_api_key.items()}
    
    return APIkeyConfig(**dict_api_key)

def build_config(args_dict=None) -> Config:
    if args_dict is None:
        args = parse_args()
        args_dict = vars(args)
    conv_args = {k.replace('c_', ''): v for k, v in args_dict.items() if k.startswith('c_')}
    retr_args = {k.replace('r_', ''): v for k, v in args_dict.items() if k.startswith('r_')}
    conv_args['model'] = sanitize_model_path(conv_args['model'])
    retr_args['model'] = sanitize_model_path(retr_args['model'])
    scrp_cfg = ScrapingConfig(
        TARGET_STATES, START_DATE, NEWS_CRAWLERS,
        N_IMG, IMG_SEARCH_FIELDS, IMG_DETAIL_FIELDS, BBOX_OFFSET
    )
    conv_cfg = ConversionConfig(**conv_args)
    retr_cfg  = RetrievalConfig(**retr_args)
    crss_cfg = CrossingConfig(US_CITIES)
    path_cfg = _compute_paths(conv_cfg, retr_cfg)
    apikey_cfg = _load_api_key(path_cfg)
    return Config(scrp=scrp_cfg, conv=conv_cfg, retr=retr_cfg, crss=crss_cfg, path=path_cfg, apikey=apikey_cfg)

if __name__ == '__main__':
    cfg = build_config()
    cfg.scrp.news_crawlers
    cfg.conv.api
    cfg.retr.model
    cfg.path.df_annotate
