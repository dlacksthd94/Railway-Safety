import json
from tqdm import tqdm
import copy
import os
from pprint import pprint
import pandas as pd
import utils
from PIL import Image
import PIL
from utils import parse_json_from_output, generate_openai, generate_hf, select_generate_func
from typing import Any

# DICT_FORM57_JSON_FORMAT = """```json
# {
#     "<field index>": {
#         "name": "<field name>",
#         "answer_type": "<free-text/digit/single choice>",
#         "choices": {
#             "<choice code>":  "<choice name>",
#         },
#     },
# }
# ```"""

DICT_FORM57_JSON_FORMAT = """```json
{
    "<field index>": {
        "name": "<field name>",
        "answer_places": {
            "<answer place name in a few words>": {
                "answer_type": "<free-text/digit/single choice>",
                "choices": {
                    "<choice code>": "<choice name>",
                },
            },
        }
    },
}
```"""

DICT_FORM57_GROUP_JSON_FORMAT = """```json
{
    "<group name>": ["<entry idx>", "<entry idx>", ...],
    "<group name>": ["<entry idx>", "<entry idx>", ...],
    ...
}
```"""

PROMPT_DICT_FORM57_TEMP = f"""
Identify the indices and names of all fields.
For each field, break down every answer place that is required to write or mark.
Strictly follow the JSON format:
{DICT_FORM57_JSON_FORMAT}
"""

# PROMPT_DICT_FORM57_TEMP = f"""
# Identify the indices and names of all fields.
# Strictly follow the JSON format:
# {DICT_FORM57_JSON_FORMAT}
# """

PROMPT_DICT_FORM57 = f"""
I'll provide you with the following:
1. The original form PDF/image.
2. A list of imperfect JSON transcription attempts of that same form.
Merge and correct them into a single transcription strictly following the JSON format:
{DICT_FORM57_JSON_FORMAT}
"""

PROMPT_DICT_FORM57_GROUP_TEMP = f"""
I'll provide you with the following:
1. The original form PDF/image.
2. An accurate transcription of the form.
Please categorize the fields into semantic groups considering the layout of the form.
Ensure that each field belongs to only one group (i.e., no overlapping groups).
The output should be in JSON format, with no additional annotations or explanations:
{DICT_FORM57_GROUP_JSON_FORMAT}
"""

PROMPT_DICT_FORM57_GROUP = f"""
I'll provide you with the following:
1. The original form PDF/image.
2. An accurate transcription of the form.
3. A list of JSON files, each containing a different grouping of fields from the same form.
Please categorize the fields into semantic groups considering the layout of the form.
Ensure that each field belongs to only one group (i.e., no overlapping groups).
The output should be in JSON format, with no additional annotations or explanations:
{DICT_FORM57_GROUP_JSON_FORMAT}
"""

def json_to_str(json_obj):
    if isinstance(json_obj, dict):
        return_str = json.dumps(json_obj, indent=4)
    elif isinstance(json_obj, list):
        return_str = '\n\n'.join(map(lambda x: json.dumps(x, indent=4), json_obj))
    else:
        raise ValueError("Unsupported JSON object type")
    return return_str

def check_dict_kv_type(dict_, k_type, v_type):
    for k, v in dict_.items():
        if not isinstance(k, k_type) or not isinstance(v, v_type):
            return False
    return True

def validate_transcription_format(dict_form57):
    """Check if the format of the transcription is valid.
    """
    if dict_form57 == {}:
        return False
    if not check_dict_kv_type(dict_form57, str, dict):
        return False
    cnt_invalid = 0
    for entry in dict_form57.values():
        if not ('name' in entry and 'answer_places' in entry and isinstance(entry['name'], str) and isinstance(entry['answer_places'], dict)):
            cnt_invalid += 1
        else:
            if not (entry['answer_places'] and check_dict_kv_type(entry['answer_places'], str, dict)):
                cnt_invalid += 1
            else:
                for ap in entry['answer_places'].values():
                    if ( (not ('answer_type' in ap and isinstance(ap['answer_type'], str)))
                        or ('choices' in ap and not (check_dict_kv_type(ap['choices'], str, str) and len(ap['choices']) != 1)) ):
                        cnt_invalid += 1
                        break
            # print(entry)
    if cnt_invalid/len(dict_form57) > 0.05:
        return False
    else:
        return True

def validate_grouping_format(dict_form57_group):
    """Check if the format of the grouping is valid.
    """
    if dict_form57_group == {}:
        return False
    if not check_dict_kv_type(dict_form57_group, str, list):
        return False
    for group in dict_form57_group.values():
        for entry_idx in group:
            if not isinstance(entry_idx, str):
                return False
    return True

def transcribe_entries(api, generator, model_path, content, n_generate, generation_config={}):
    generate_func = select_generate_func(api)
    
    list_response = []
    for i in range(n_generate):
        while True:
            with utils.Timer(f'{i}\t' + model_path):
                output = generate_func(generator, model_path, content, generation_config)
            dict_form57 = parse_json_from_output(output)

            if validate_transcription_format(dict_form57):
                break
        
        list_response.append(dict_form57)
    return list_response

def group_entries(api, generator, model_path, content, n_generate, generation_config={}):
    generate_func = select_generate_func(api)
    
    list_response = []
    for i in range(n_generate):
        while True:
            with utils.Timer(f'{i}\t' + model_path):
                output = generate_func(generator, model_path, content, generation_config)
            dict_form57_group = parse_json_from_output(output)

            group_overlap = False
            all_entry_idx = []
            for _, list_entry_idx in dict_form57_group.items():
                for entry_idx in list_entry_idx:
                    if entry_idx not in all_entry_idx:
                        all_entry_idx.append(entry_idx)
                    else:
                        group_overlap = True
                        break
            if dict_form57_group and not group_overlap:
                break
        
        list_response.append(dict_form57_group)
    return list_response

def csv_to_json(cfg):
    
    if os.path.exists(cfg.path.form57_json):
        with open(cfg.path.form57_json, 'r') as f:
            dict_form57 = json.load(f)

    else:
        df_data = pd.read_csv(cfg.path.df_record)
        df_data = df_data[df_data['State Name'].str.title().isin(cfg.scrp.target_states)]
        df_data['Date'] = pd.to_datetime(df_data['Date'])
        df_data = df_data[df_data['Date'] >= cfg.scrp.start_date]

        list_col = df_data.columns.tolist()

        ############ Match the columns containing codes with the columns containing labels.
        dict_code_label = {}
        for i, col in enumerate(list_col):
            if col.endswith('Code'):
                col_code = col
                col_label = list_col[i + 1]
                dict_code_label[col_code] = col_label
        
        ############ Extract the options (multiple choices), if any
        dict_entry_choice = {}
        for col_code, col_label in dict_code_label.items():
            df_drop_na = df_data[[col_code, col_label]].dropna()
            df_astype = df_drop_na
            df_astype[col_code] = df_astype[col_code].apply(lambda x: int(x) if isinstance(x, float) else x)
            df_astype[col_code] = df_astype[col_code].astype(str)
            df_drop_dup = df_astype.drop_duplicates()
            df_sort = df_drop_dup.sort_values(col_code)
            
            dict_entry_choice_temp = df_sort.set_index(col_code)[col_label].to_dict()
            if len(dict_entry_choice_temp) <= 15:
                dict_entry_choice[col_label] = dict_entry_choice_temp

        ############### manually add the options for special cases
        for i, col in enumerate(list_col):
            if col in ['Warning Connected To Signal', 'Crossing Illuminated', 'User Struck By Second Train', 'Driver Passed Vehicle']:
                dict_entry_choice[col] = {'1': 'Yes', '2': 'No', '3': 'Unknown'}
            elif col == 'User Sex':
                dict_entry_choice[col] = {'1': 'Male', '2': 'Female'}
            elif col == 'Driver In Vehicle':
                dict_entry_choice[col] = {'1': 'Yes', '2': 'No'}
            elif col == 'AM/PM':
                dict_entry_choice[col] = {'1': 'AM', '2': 'PM'}
            elif col == 'Estimated/Recorded Speed':
                dict_entry_choice[col] = {'1': 'R', '2': 'E'}

        ############ Convert into JSON
        list_col_wo_code = list(filter(lambda col: not col.endswith('Code'), list_col))
        list_not_use = list(range(53, 79)) + list(range(106, 130))
        dict_form57 = {str(i): {'name': col} for i, col in enumerate(list_col_wo_code) if i not in list_not_use}
        for i, dict_meta_info in dict_form57.items():
            col = dict_meta_info['name']
            dict_meta_info['answer_places'] = {col: {}}
            if col in dict_entry_choice:
                choices = dict_entry_choice[col]
                dict_meta_info['answer_places'][col]['answer_type'] = 'single choice'
                dict_meta_info['answer_places'][col]['choices'] = choices
            elif pd.api.types.is_numeric_dtype(df_data[col]):
                dict_meta_info['answer_places'][col]['answer_type'] = 'digit'
            else:
                dict_meta_info['answer_places'][col]['answer_type'] = 'free-text'

        with open(cfg.path.form57_json, 'w') as f:
            json.dump(dict_form57, f, indent=4)
                
    return dict_form57

def pdf_to_json(cfg):
    dict_form57_group = None
    api, model_path, n_generate, _ = cfg.conv.to_tuple()
    ############# Google Document AI API

    if api == 'Google_DocAI':
        from google.cloud import documentai_v1 as documentai # type: ignore

        dict_type_pid = {
            'form_parser': 'b4aa30ed34270c72',
            'layout_parser': '489307cad77319fb'
        }
        
        # 1) Configure client
        project_id = "railway-safety-460721"
        location   = "us"                # e.g. "us", "eu"
        processor_id = dict_type_pid[model_path]
        client = documentai.DocumentProcessorServiceClient(
            client_options={"api_endpoint": f"{location}-documentai.googleapis.com"}
        )

        # 2) Read your PDF
        with open(cfg.path.form57_pdf, "rb") as f:
            pdf_bytes = f.read()

        # 3) Build the request
        name = client.processor_path(project_id, location, processor_id)
        request = documentai.ProcessRequest(
            name=name,
            raw_document=documentai.RawDocument(content=pdf_bytes, mime_type="application/pdf")
        )

        # 4) Send to API
        result = client.process_document(request=request)
        document = result.document

        document.text

        # ####### vision API (OCR) instead of document AI
        # # https://cloud.google.com/vision/docs/pdf#vision_text_detection_pdf_gcs-python
        # from google.cloud import vision

        # # 1) Configure client
        # client_options = {
        #     "api_endpoint": "us-vision.googleapis.com"
        # }
        # client = vision.ImageAnnotatorClient(client_options=client_options)

        # # 2) Read your PDF
        # with open(path_img_form57, "rb") as f:
        #     img_bytes = f.read()

        dict_form57 = None
        
    ############# AWS Textract API
    # reference
    # https://docs.aws.amazon.com/textract/latest/dg/API_AnalyzeDocument.html
    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/textract/client/analyze_document.html
    # https://docs.aws.amazon.com/textract/latest/dg/analyzing-document-text.html?utm_source=chatgpt.com
    elif api == 'AWS_Textract':
        import boto3
        import io
        from PIL import Image, ImageDraw
        from pdf2image import convert_from_path

        def analyze_pdf(path, client, feature_type):
            with open(path, 'rb') as f:
                pdf_bytes = f.read()

            resp = client.analyze_document(
                Document={'Bytes': pdf_bytes},
                FeatureTypes=feature_type
            )
            return resp

        def analyze_img(path, clien, feature_type):
            with open(path, 'rb') as f:
                img_bytes = f.read()

            resp = client.analyze_document( # type: ignore
                Document={'Bytes': img_bytes},
                FeatureTypes=feature_type
            )
            return resp

        def ShowBoundingBox(draw,box,width,height,boxColor,linewidth=1):
            left = width * box['Left']
            top = height * box['Top']
            draw.rectangle([left,top, left + (width * box['Width']), top +(height * box['Height'])],outline=boxColor, width=linewidth)

        def ShowSelectedElement(draw,box,width,height,boxColor,linewidth=1):
            left = width * box['Left']
            top = height * box['Top']
            draw.rectangle([left,top, left + (width * box['Width']), top +(height * box['Height'])],fill=boxColor, width=linewidth)

        def map_id_block(blocks):
            dict_id_block = {}
            for block in tqdm(blocks):
                if block['Id'] not in dict_id_block:
                    dict_id_block[block['Id']] = block
                else:
                    break
            assert len(dict_id_block) == len(blocks)
            return dict_id_block

        def draw_boxes(blocks, path_form57):
            # if path_form57.endswith('pdf'):
            images = convert_from_path(path_form57)
            image = images[0]
            # elif path_form57.endswith('jpg'):
            #     image = Image.open(path_form57)
            width, height = image.size

            dict_id_block = map_id_block(blocks)
            for block in blocks:
                draw=ImageDraw.Draw(image)

                if block['BlockType'] == "KEY_VALUE_SET":
                    if block['EntityTypes'][0] == "KEY":
                        ShowBoundingBox(draw, block['Geometry']['BoundingBox'],width,height,'red')
                    else:
                        ShowBoundingBox(draw, block['Geometry']['BoundingBox'],width,height,'green')
                
                if block['BlockType'] == 'TABLE':
                    ShowBoundingBox(draw, block['Geometry']['BoundingBox'],width,height, 'red', linewidth=4)
                if block['BlockType'] == 'MERGED_CELL':
                    ShowBoundingBox(draw, block['Geometry']['BoundingBox'],width,height, 'blue', linewidth=2)
                if block['BlockType'] == 'CELL':
                    ShowBoundingBox(draw, block['Geometry']['BoundingBox'],width,height, 'orange')
                if block['BlockType'] == 'TABLE_FOOTER':
                    ShowBoundingBox(draw, block['Geometry']['BoundingBox'],width,height, 'green')
                
                if block['BlockType'] == 'LAYOUT_HEADER':
                    ShowBoundingBox(draw, block['Geometry']['BoundingBox'],width,height, 'green')
                if block['BlockType'] == 'LAYOUT_TITLE':
                    ShowBoundingBox(draw, block['Geometry']['BoundingBox'],width,height, 'orange')
                if block['BlockType'] == 'LAYOUT_KEY_VALUE':
                    ShowBoundingBox(draw, block['Geometry']['BoundingBox'],width,height, 'red', linewidth=2)
                    for id in block['Relationships'][0]['Ids']:
                        block_line = dict_id_block[id]
                        ShowBoundingBox(draw, block_line['Geometry']['BoundingBox'],width,height, 'blue')
                if block['BlockType'] == 'LAYOUT_FOOTER':
                    ShowBoundingBox(draw, block['Geometry']['BoundingBox'],width,height, 'lightgreen')
                
                if block['BlockType'] == 'SELECTION_ELEMENT':
                    if block['SelectionStatus'] =='SELECTED':
                        ShowSelectedElement(draw, block['Geometry']['BoundingBox'],width,height, 'blue')
            image.save(cfg.path.form57_annotated)

        session = boto3.Session(profile_name='textract')
        client = session.client(service_name='textract', region_name='us-west-2')
        feature_type = ['TABLES','FORMS','SIGNATURES', 'LAYOUT']
        # feature_type = ['LAYOUT']
        # feature_type = ['TABLES']
        feature_type = ['FORMS']
        # feature_type = ['QUERIES'] # must use queries config

        result_pdf = analyze_pdf(cfg.path.form57_pdf, client, feature_type)
        blocks = result_pdf['Blocks']
        draw_boxes(blocks, cfg.path.form57_pdf)
        geometry_popped = [block.pop('Geometry') for block in blocks]
        # result_pdf['Blocks'] = [block for block in blocks if block['BlockType'] != 'WORD']
        {block['BlockType'] for block in blocks}
        with open(cfg.path.form57_json, 'w') as f:
            json.dump(result_pdf, f, indent=4)

        # make tree
        dict_id_block = map_id_block(blocks)
        tree = copy.deepcopy(blocks[0])
        tree['Relationships'][0]['Children'] = {}
        for block in tqdm(blocks[1:]):
            id = block['Id']
            if block['BlockType'] in ['LAYOUT_KEY_VALUE', 'TABLE']:
                dict_children = {}
                for id_child in block['Relationships'][0]['Ids']:
                    dict_children[id_child] = dict_id_block[id_child]
                block['Relationships'][0]['Children'] = dict_children
                tree['Relationships'][0]['Children'][id] = block
            elif block['BlockType'] not in ['WORD', 'SELECTION_ELEMENT', 'CELL', 'MERGED_CELL', 'TABLE_FOOTER']:
                tree['Relationships'][0]['Ids'].remove(id)

        # images = convert_from_path(cfg.path.form57_pdf)
        # images[0].save(path_img_form57, "JPEG")
        # result_img = analyze_img(path_img_form57, client, feature_type)

        # len(result_pdf['Blocks'])
        # len(result_img['Blocks'])

        dict_form57 = None

    ############# Azure Form Recognizer API
    # https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/concept/choose-model-feature?view=doc-intel-4.0.0#pretrained-document-analysis-models
    elif api == 'Azure_FormRecognizer':
        from azure.core.credentials import AzureKeyCredential # type: ignore
        from azure.ai.documentintelligence import DocumentIntelligenceClient # type: ignore
        from azure.ai.documentintelligence.models import AnalyzeResult, AnalyzeDocumentRequest # type: ignore

        # set `<your-endpoint>` and `<your-key>` variables with the values from the Azure portal
        endpoint = 'https://railway-safety.cognitiveservices.azure.com/'
        key = '7AP8Y2CeuNFAIQtxolqWXi4t45gFBfjc1l57heQalSQb0Y4qqjd0JQQJ99BFAC4f1cMXJ3w3AAALACOGmlw1'

        def get_words(page, line):
            result = []
            for word in page.words:
                if _in_span(word, line.spans):
                    result.append(word)
            return result

        def _in_span(word, spans):
            for span in spans:
                if word.span.offset >= span.offset and (
                    word.span.offset + word.span.length
                ) <= (span.offset + span.length):
                    return True
            return False

        with open(cfg.path.form57_pdf, "rb") as f:
            byte_pdf = f.read()

        document_intelligence_client = DocumentIntelligenceClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )

        poller = document_intelligence_client.begin_analyze_document(
            "prebuilt-layout", AnalyzeDocumentRequest(bytes_source=byte_pdf
        ))

        result: AnalyzeResult = poller.result()

        if result.styles and any([style.is_handwritten for style in result.styles]):
            print("Document contains handwritten content")
        else:
            print("Document does not contain handwritten content")

        # for page in result.pages:
        #     print(f"----Analyzing layout from page #{page.page_number}----")
        #     print(
        #         f"Page has width: {page.width} and height: {page.height}, measured with unit: {page.unit}"
        #     )

        #     if page.lines:
        #         for line_idx, line in enumerate(page.lines):
        #             words = get_words(page, line)
        #             print(
        #                 f"...Line # {line_idx}: '{line.content}' "
        #                 # f"...Line # {line_idx} has word count {len(words)} and text '{line.content}' "
        #                 # f"within bounding polygon '{line.polygon}'"
        #             )

        #             # for word in words:
        #             #     print(
        #             #         f"......Word '{word.content}' has a confidence of {word.confidence}"
        #             #     )

        #     if page.selection_marks:
        #         for selection_mark in page.selection_marks:
        #             print(
        #                 f"Selection mark is '{selection_mark.state}' within bounding polygon "
        #                 f"'{selection_mark.polygon}' and has a confidence of {selection_mark.confidence}"
        #             )

        if result.tables:
            for table_idx, table in enumerate(result.tables):
                print(
                    f"Table # {table_idx} has {table.row_count} rows and {table.column_count} columns"
                )
                # if table.bounding_regions:
                #     for region in table.bounding_regions:
                #         print(
                #             f"Table # {table_idx} location on page: {region.page_number} is {region.polygon}"
                #         )
                for cell in table.cells:
                    print(
                        f"...Cell[{cell.row_index}][{cell.column_index}] has text '{cell.content}'"
                    )
                    # if cell.bounding_regions:
                    #     for region in cell.bounding_regions:
                    #         print(
                    #             f"...content on page {region.page_number} is within bounding polygon '{region.polygon}'"
                    #         )

        print("----------------------------------------")

        # +++ draw a bounding box around the table and save as an image

        dict_form57 = None

    ############# OpenAI API
    elif api == 'OpenAI':
        from openai import OpenAI

        API_key = cfg.apikey.openai
        client = OpenAI(api_key=API_key)

        with open(cfg.path.form57_pdf, "rb") as f:
            pdf_file = client.files.create(
                file=f,
                purpose="user_data"
            )

        ############ transcribe
        if os.path.exists(cfg.path.form57_json):
            with open(cfg.path.form57_json, 'r') as f:
                dict_form57 = json.load(f)

        else:
            ############ transcribe the form N times
            content = [
                {"type": "input_text", "text": PROMPT_DICT_FORM57_TEMP},
                {"type": "input_file", "file_id": pdf_file.id},
            ]
            list_dict_form57_temp = transcribe_entries(api, client, model_path, content, n_generate, generation_config=None)

            ############ merge the transcripts into one
            content = [
                {"type": "input_file", "file_id": pdf_file.id},
                {"type": "input_text", "text": PROMPT_DICT_FORM57},
                {"type": "input_text", "text": json_to_str(list_dict_form57_temp)}
            ]
            list_dict_form57 = transcribe_entries(api, client, model_path, content, n_generate=1, generation_config=None)
            dict_form57 = list_dict_form57[0]
            
            with open(cfg.path.form57_json, 'w') as f:
                json.dump(dict_form57, f, indent=4)
        
        ############ group the entries
        if os.path.exists(cfg.path.form57_json_group):
            with open(cfg.path.form57_json_group, 'r') as f:
                dict_form57_group = json.load(f)
                
        else:
            ############ group the entries N times
            content = [
                {"type": "input_text", "text": PROMPT_DICT_FORM57_GROUP_TEMP},
                {"type": "input_file", "file_id": pdf_file.id},
                {"type": "input_text", "text": json_to_str(dict_form57)}
            ]
            list_dict_form57_group_temp = group_entries(api, client, model_path, content, n_generate, generation_config=None)
            
            ############ merge groupings into one
            content = [
                {"type": "input_file", "file_id": pdf_file.id},
                {"type": "input_text", "text": PROMPT_DICT_FORM57_GROUP},
                {"type": "input_text", "text": json_to_str(dict_form57)},
                {"type": "input_text", "text": json_to_str(list_dict_form57_group_temp)}
            ]
            list_dict_form57_group = group_entries(api, client, model_path, content, n_generate=1, generation_config=None)
            dict_form57_group = list_dict_form57_group[0]

            with open(cfg.path.form57_json_group, 'w') as f:
                json.dump(dict_form57_group, f, indent=4)
    
    else:
        raise ValueError(f"Unsupported API: {api}")

    return dict_form57, dict_form57_group

def img_to_json(cfg):
    
    dict_form57_group = None
    api, model_path, n_generate, _ = cfg.conv.to_tuple()
    image = Image.open(cfg.path.form57_img)

    if api == 'Huggingface':
        import torch
        import gc
        from transformers import pipeline

        dict_model_config = {
            # 'Qwen/Qwen2.5-VL-7B-Instruct': {}, # good
            'Qwen/Qwen2.5-VL-32B-Instruct': {},
            'Qwen/Qwen2.5-VL-72B-Instruct': {'load_in_4bit': True},
            # 'microsoft/GUI-Actor-7B-Qwen2.5-VL': {}, # bad
            # 'OpenGVLab/InternVL3-8B': {}, #must be used with custom code to correctly load the model
            # 'OpenGVLab/InternVL3-8B-Instruct': {}, #must be used with custom code to correctly load the model
            # 'OpenGVLab/InternVL3-8B-hf': {}, # good
            # 'OpenGVLab/InternVL3-14B-hf': {},
            'OpenGVLab/InternVL3-38B-hf': {},
            'OpenGVLab/InternVL3-78B-hf': {'load_in_8bit': True},
            'OpenGVLab/InternVL3_5-38B-HF': {},
            # 'microsoft/Phi-3.5-vision-instruct': {}, #error
            # 'google/gemma-3-4b-it': {}, # bad
            'google/gemma-3-12b-it': {},
            'google/gemma-3-27b-it': {},
            # 'microsoft/OmniParser-v2.0': {}, #error
            # 'U4R/StructTable-base': {}, #StructEqTable. cannot use chat template input
            # 'U4R/StructTable-InternVL2-1B': {}, #StructEqTable. must be used with custom code to correctly load the model
            # 'stepfun-ai/GOT-OCR-2.0-hf': {}, #cannot use chat template input
            ### https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md
            # 'llava-hf/llava-1.5-7b-hf': {}, # bad
            'llava-hf/llava-1.5-13b-hf': {'load_in_8bit': True},
            # 'llava-hf/vip-llava-7b-hf': {}, # bad
            'llava-hf/vip-llava-13b-hf': {'load_in_8bit': True},
            # 'llava-hf/llava-v1.6-vicuna-7b-hf': {}, # bad
            # 'llava-hf/llava-v1.6-mistral-7b-hf': {}, # bad
            'llava-hf/llava-v1.6-34b-hf': {'dtype': torch.float16},
            # 'llava-hf/llava-interleave-qwen-7b-hf': {}, # bad
            # 'llava-hf/llama3-llava-next-8b-hf': {}, # bad
            # 'allenai/olmOCR-7B-0225-preview': {}, #olmOCR(OLMo OCR)
            # 'meta-llama/Llama-3.2-11B-Vision': {}, # cannot use chat template input
            # 'meta-llama/Llama-3.2-11B-Vision-Instruct': {}, # bad
            # 'nvidia/Eagle2.5-8B': {}, # must be used with custom code to correctly load the model
            'ByteDance-Seed/UI-TARS-1.5-7B': {}, # good
            # 'ByteDance/Sa2VA-26B': {'load_in_8bit': True}, # must be used with custom code to correctly load the model
            # 'allenai/Molmo-7B-D-0924': {}, # must be used with custom code to correctly load the model
            # 'microsoft/layoutlmv3-large': {}, # maximum input size is 512 tokens / needs OCR / focused on small-sized receipts and invoices
        }

        quant_config = dict_model_config[model_path]
        generation_config_base = {'max_new_tokens': 8192}
        generation_config_sample = {**generation_config_base, 'do_sample': True}#, 'temperature': 1, 'top_p': 0.95} # sample or beam sample
        generation_config_search = {**generation_config_base, 'do_sample': False} # greedy search or beam search

        pipe = pipeline(model=model_path, device_map='auto', model_kwargs=quant_config) # type: ignore

        ############### transcribe
        path_form57_json_temp = cfg.path.form57_json.replace('.json', '') + '_temp.json'
        
        if os.path.exists(path_form57_json_temp):
            with open(path_form57_json_temp, 'r') as f:
                list_dict_form57_temp = json.load(f)

        else:
            ############ transcribe the form N times
            content = [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT_DICT_FORM57_TEMP},
            ]
            list_dict_form57_temp = transcribe_entries(api, pipe, model_path, content, n_generate, generation_config_sample)

            with open(path_form57_json_temp, 'w') as f:
                json.dump(list_dict_form57_temp, f, indent=4)
            
        if os.path.exists(cfg.path.form57_json):
            with open(cfg.path.form57_json, 'r') as f:
                dict_form57 = json.load(f)
        
        else:
            ############ merge the transcripts into one
            content = [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT_DICT_FORM57},
                {"type": "text", "text": json_to_str(list_dict_form57_temp)}
            ]

            list_dict_form57 = transcribe_entries(api, pipe, model_path, content, n_generate=1, generation_config=generation_config_search)
            dict_form57 = list_dict_form57[0]

            with open(cfg.path.form57_json, 'w') as f:
                json.dump(dict_form57, f, indent=4)

        ############ group the entries
        path_form57_json_group_temp = cfg.path.form57_json_group.replace('.json', '') + '_temp.json'

        if os.path.exists(path_form57_json_group_temp):
            with open(path_form57_json_group_temp, 'r') as f:
                list_dict_form57_group_temp = json.load(f)
                
        else:
            ############ categorize the entries
            content = [
                { "type": "text", "text": PROMPT_DICT_FORM57_GROUP_TEMP},
                { "type": "image", "image": image},
                { "type": "text", "text": json_to_str(dict_form57)}
            ]
            list_dict_form57_group_temp = group_entries(api, pipe, model_path, content, n_generate, generation_config_sample)

            with open(path_form57_json_group_temp, 'w') as f:
                json.dump(list_dict_form57_group_temp, f, indent=4)
            
        if os.path.exists(cfg.path.form57_json_group):
            with open(cfg.path.form57_json_group, 'r') as f:
                dict_form57_group = json.load(f)
        
        else:
            ############ merge groupings into one
            content = [
                { "type": "image", "image": image},
                { "type": "text", "text": PROMPT_DICT_FORM57_GROUP},
                { "type": "text", "text": json.dumps(dict_form57, indent=4)},
                { "type": "text", "text": json_to_str(list_dict_form57_group_temp)}
            ]
            list_dict_form57_group = group_entries(api, pipe, model_path, content, n_generate=1, generation_config=generation_config_search)
            dict_form57_group = list_dict_form57_group[0]

            with open(cfg.path.form57_json_group, 'w') as f:
                json.dump(dict_form57_group, f, indent=4)
            
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
    
    elif api == 'OpenAI':
        from openai import OpenAI

        API_key = cfg.apikey.openai
        client = OpenAI(api_key=API_key)

        with open(cfg.path.form57_img, "rb") as f:
            img_file = client.files.create(
                file=f,
                purpose="vision"
            )

        ############ transcribe
        path_form57_json_temp = cfg.path.form57_json.replace('.json', '') + '_temp.json'
        
        if os.path.exists(path_form57_json_temp):
            with open(path_form57_json_temp, 'r') as f:
                list_dict_form57_temp = json.load(f)

        else:
            ############ transcribe the form N times
            content = [
                {"type": "input_text", "text": PROMPT_DICT_FORM57_TEMP},
                {"type": "input_image", "file_id": img_file.id},
            ]
            list_dict_form57_temp = transcribe_entries(api, client, model_path, content, n_generate, generation_config=None)

            with open(path_form57_json_temp, 'w') as f:
                json.dump(list_dict_form57_temp, f, indent=4)
            
        if os.path.exists(cfg.path.form57_json):
            with open(cfg.path.form57_json, 'r') as f:
                dict_form57 = json.load(f)
            
        else:
            ############ merge the transcripts into one
            content = [
                {"type": "input_image", "file_id": img_file.id},
                {"type": "input_text", "text": PROMPT_DICT_FORM57},
                {"type": "input_text", "text": json_to_str(list_dict_form57_temp)}
            ]
            list_dict_form57 = transcribe_entries(api, client, model_path, content, n_generate=1, generation_config=None)
            dict_form57 = list_dict_form57[0]
            
            with open(cfg.path.form57_json, 'w') as f:
                json.dump(dict_form57, f, indent=4)
        
        ############ group the entries
        path_form57_json_group_temp = cfg.path.form57_json_group.replace('.json', '') + '_temp.json'

        if os.path.exists(path_form57_json_group_temp):
            with open(path_form57_json_group_temp, 'r') as f:
                list_dict_form57_group_temp = json.load(f)
                
        else:
            ############ group the entries N times
            content = [
                {"type": "input_text", "text": PROMPT_DICT_FORM57_GROUP_TEMP},
                {"type": "input_image", "file_id": img_file.id},
                {"type": "input_text", "text": json_to_str(dict_form57)}
            ]
            list_dict_form57_group_temp = group_entries(api, client, model_path, content, n_generate, generation_config=None)

            with open(path_form57_json_group_temp, 'w') as f:
                json.dump(list_dict_form57_group_temp, f, indent=4)
        
        if os.path.exists(cfg.path.form57_json_group):
            with open(cfg.path.form57_json_group, 'r') as f:
                dict_form57_group = json.load(f)
        
        else:
            ############ merge groupings into one
            content = [
                {"type": "input_image", "file_id": img_file.id},
                {"type": "input_text", "text": PROMPT_DICT_FORM57_GROUP},
                {"type": "input_text", "text": json_to_str(dict_form57)},
                {"type": "input_text", "text": json_to_str(list_dict_form57_group_temp)}
            ]
            list_dict_form57_group = group_entries(api, client, model_path, content, n_generate=1, generation_config=None)
            dict_form57_group = list_dict_form57_group[0]

            with open(cfg.path.form57_json_group, 'w') as f:
                json.dump(dict_form57_group, f, indent=4)
    
    else:
        raise ValueError(f"Unsupported API: {api}")

    return dict_form57, dict_form57_group

def convert_to_json(cfg) -> tuple[dict[str, dict[str, Any]] | None, dict[str, list[int]] | None]:
    if cfg.conv.json_source == 'csv':
        dict_form57 = csv_to_json(cfg)
        dict_form57_group = None
    elif cfg.conv.json_source == 'pdf':
        dict_form57, dict_form57_group = pdf_to_json(cfg)
    elif cfg.conv.json_source == 'img':
        dict_form57, dict_form57_group = img_to_json(cfg)
    else:
        dict_form57, dict_form57_group = None, None
    return dict_form57, dict_form57_group

if __name__ == '__main__':
    # dict_form57 = parse_json_from_output(output)
    # validate_transcription_format(dict_form57)
    
    answer_format = """```
    "answer_places": {
        "<answer place name>": {
            "type": "<free-text/digit/single choice/multiple choice>",
            "choices": {
                "<choice code>": "<choice name>",
            },
        },
    }
    ```"""
    target_field = '41. Highway User'
    target_field = '43. View of Track Obscured by (primary obstruction)'
    target_field = '52. Passengers on Train'
    prompt = f"""Within the field "{target_field}", break down every answer place that is required to write or mark.
    Strictly follow the JSON format:
    {answer_format}
    """
    # with utils.Timer(model_path):
    #     # print(generate_hf(pipe, model_path, [{"type": "image", "image": image}, {"type": "text", "text": prompt},], generation_config_search)) # HF
    #     print(generate_openai(client, model_path, [{"type": "input_image", "file_id": img_file.id}, {"type": "input_text", "text": prompt},])) # OpenAI
    