import json
from tqdm import tqdm
import copy
import os
import pprint
import pandas as pd
import utils_scrape
import time
import utils

PROMPT_DICT_FORM57_TEMP = """
Transcribe all fields in JSON format:
```json
{
    "<field_id>": {
        "name": "<field label>",
        "choices": {
            "<choice_key>": "<choice_label>",
        }
    },
}
```
"""

PROMPT_DICT_FORM57 = f"""
I'll give you:
1. The original form PDF.
2. A list of JSON files, each one an imperfect transcription of that same form.
Please merge them into a single, accurate transcription.
"""

PROMPT_DICT_FORM57_GROUP_TEMP = """
I'll give you:
1. The original form PDF.
2. A single, accurate transcription of the PDF.
Some entries may be ambiguous on their own, but grouping semantically related entries together can help clarify their meaning.
Please categorize them accordingly.
The output should be in JSON format without annotations or explanations:
```
{
    group: [entry, entry, ...],
    group: [entry, entry, ...],
    ...
}
```
"""

PROMPT_DICT_FORM57_GROUP = f"""
I'll give you:
1. The original form PDF.
2. A single, accurate transcription of the PDF.
3. A list of JSON files, each containing a different grouping of entries from the same form.
Some entries may be ambiguous on their own, but grouping semantically related entries together can help clarify their meaning.
Please merge them into a single, unified grouping without annotations or explanations.
"""

def parse_json_from_output(output):
    try:
        if '```' in output:
            json_start_index = output.index('```')
            json_end_index = output.rindex('```')
            str_form57 = output[json_start_index:json_end_index].strip('`')
            if str_form57.startswith('json'):
                str_form57 = str_form57.replace('json', '', 1)
            dict_form57 = json.loads(str_form57)
        else:
            dict_form57 = json.loads(output)
    except:
        dict_form57 = ''
    return dict_form57

def csv_to_json(path_form57_csv, path_form57_json):
    
    if os.path.exists(path_form57_json):
        with open(path_form57_json, 'r') as f:
            dict_form57 = json.load(f)

    else:
        df_data = pd.read_csv(path_form57_csv)
        # df_data = df_data[df_data['State Name'] == 'CALIFORNIA']
        df_data['hash_id'] = df_data.apply(utils_scrape.hash_row, axis=1)
        df_data['Date'] = pd.to_datetime(df_data['Date'])
        df_data = df_data[df_data['Date'] >= '2000-01-01']

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
        pprint(dict_entry_choice)

        ############ Convert into JSON
        list_col_wo_code = list(filter(lambda col: not col.endswith('Code'), list_col))
        list_col_wo_code.remove('hash_id')
        dict_form57 = {i: {'name': col} for i, col in enumerate(list_col_wo_code)}
        for i, dict_meta_info in dict_form57.items():
            col = dict_meta_info['name']
            if col in dict_entry_choice:
                choices = dict_entry_choice[col]
                dict_form57[i]['choices'] = choices
        pprint(dict_form57)

        with open(path_form57_json, 'w') as f:
            json.dump(dict_form57, f, indent=4)
                
    return dict_form57

def pdf_to_json(path_form57_pdf, path_form57_json, path_form57_json_group, config_conversion):
    dict_form57_group = None
    api, model, n_generate, _ = config_conversion.to_tuple()
    ############# Google Document AI API

    if api == 'Google_DocAI':
        from google.cloud import documentai_v1 as documentai

        dict_type_pid = {
            'form_parser': 'b4aa30ed34270c72',
            'layout_parser': '489307cad77319fb'
        }
        
        # 1) Configure client
        project_id = "railway-safety-460721"
        location   = "us"                # e.g. "us", "eu"
        processor_id = dict_type_pid[model]
        client = documentai.DocumentProcessorServiceClient(
            client_options={"api_endpoint": f"{location}-documentai.googleapis.com"}
        )

        # 2) Read your PDF
        with open(path_form57_pdf, "rb") as f:
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
        
    ############# AWS Textract API
    # reference
    # https://docs.aws.amazon.com/textract/latest/dg/API_AnalyzeDocument.html
    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/textract/client/analyze_document.html
    # https://docs.aws.amazon.com/textract/latest/dg/analyzing-document-text.html?utm_source=chatgpt.com
    if api == 'AWS_Textract':
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

            resp = client.analyze_document(
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
            if path_form57.endswith('pdf'):
                images = convert_from_path(path_form57)
                image = images[0]
            elif path_form57.endswith('jpg'):
                image = Image.open(path_form57)
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
            image.save(DATA_FOLDER + 'annotated_form.png')

        session = boto3.Session(profile_name='textract')
        client = session.client(service_name='textract', region_name='us-west-2')
        feature_type = ['TABLES','FORMS','SIGNATURES', 'LAYOUT']
        # feature_type = ['LAYOUT']
        # feature_type = ['TABLES']
        feature_type = ['FORMS']
        # feature_type = ['QUERIES'] # must use queries config

        result_pdf = analyze_pdf(path_form57_pdf, client, feature_type)
        blocks = result_pdf['Blocks']
        draw_boxes(blocks, path_form57_pdf)
        geometry_popped = [block.pop('Geometry') for block in blocks]
        # result_pdf['Blocks'] = [block for block in blocks if block['BlockType'] != 'WORD']
        {block['BlockType'] for block in blocks}
        with open(path_form57_json, 'w') as f:
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

        # images = convert_from_path(path_form57_pdf)
        # images[0].save(path_img_form57, "JPEG")
        # result_img = analyze_img(path_img_form57, client, feature_type)

        # len(result_pdf['Blocks'])
        # len(result_img['Blocks'])

    ############# Azure Form Recognizer API
    # https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/concept/choose-model-feature?view=doc-intel-4.0.0#pretrained-document-analysis-models
    if api == 'Azure_FormRecognizer':
        from azure.core.credentials import AzureKeyCredential
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.ai.documentintelligence.models import AnalyzeResult
        from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

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

        with open(path_form57_pdf, "rb") as f:
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

    ############# OpenAI API
    if api == 'OpenAI':
        from openai import OpenAI

        API_key = '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'
        client = OpenAI(api_key=API_key)

        with open(path_form57_pdf, "rb") as f:
            pdf_file = client.files.create(
                file=f,
                purpose="user_data"
            )

        ############ transcribe
        if os.path.exists(path_form57_json):
            with open(path_form57_json, 'r') as f:
                dict_form57 = json.load(f)

        else:
            ############ transcribe pdf N times
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            { "type": "text", "text": PROMPT_DICT_FORM57_TEMP},
                            { "type": "file", "file": {"file_id": pdf_file.id} },
                        ]
                    },
                ],
                n=n_generate,
                # seed=0,
                # temperature=0,
            )

            list_dict_form57_temp = []
            for choice in response.choices:
                output = choice.message.content
                dict_form57_temp = parse_json_from_output(output)
                list_dict_form57_temp.append(dict_form57_temp)

            ############ merge transcripts into one
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            { "type": "input_file", "file_id": pdf_file.id },
                            { "type": "input_text", "text": PROMPT_DICT_FORM57},
                            { "type": "input_text", "text": str(list_dict_form57_temp)}
                        ]
                    }
                ],
            )
            # print(response.output_text)

            output = response.output_text
            dict_form57 = parse_json_from_output(output)

            with open(path_form57_json, 'w') as f:
                json.dump(dict_form57, f, indent=4)
        
        ############ categorize the entries
        if os.path.exists(path_form57_json_group):
            with open(path_form57_json_group, 'r') as f:
                dict_form57_group = json.load(f)
                
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": PROMPT_DICT_FORM57_GROUP_TEMP},
                        { "type": "file", "file": {"file_id": pdf_file.id} },
                        { "type": "text", "text": str(dict_form57)}
                    ]
                },
            ],
            with utils.Timer(path_form57_json):
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    n=n_generate,
                    # seed=0,
                    # temperature=0,
                )

            list_dict_form57_group_temp = []
            for choice in response.choices:
                output = choice.message.content
                dict_form57_group_temp = parse_json_from_output(output)
                list_dict_form57_group_temp.append(dict_form57_group_temp)

            ############ merge groupings into one
            messages = [
                {
                    "role": "user",
                    "content": [
                        { "type": "input_file", "file_id": pdf_file.id },
                        { "type": "input_text", "text": PROMPT_DICT_FORM57_GROUP},
                        { "type": "input_text", "text": str(dict_form57)},
                        { "type": "input_text", "text": str(list_dict_form57_group_temp)}
                    ]
                }
            ]
            with utils.Timer(path_form57_json):
                response = client.responses.create(
                    model=model,
                    input=messages
                )

            output = response.output_text
            dict_form57_group = parse_json_from_output(output)
            
            with open(path_form57_json_group, 'w') as f:
                json.dump(dict_form57_group, f, indent=4)

    return dict_form57, dict_form57_group

def img_to_json(path_form57_img, path_form57_json, path_form57_json_group, config_conversion):
    
    dict_form57_group = None
    api, model, n_generate, _ = config_conversion.to_tuple()

    if api == 'Huggingface':
        import torch
        import gc
        from transformers import pipeline
        from PIL import Image

        dict_model_config = {
            'Qwen/Qwen2.5-VL-7B-Instruct': {'num_beams': 1}, # good
            # 'microsoft/GUI-Actor-7B-Qwen2.5-VL': {}, # bad
            # 'OpenGVLab/InternVL3-8B': {}, #must be used with custom code to correctly load the model
            # 'OpenGVLab/InternVL3-8B-Instruct': {}, #must be used with custom code to correctly load the model
            'OpenGVLab/InternVL3-8B-hf': {'num_beams': 1}, # good
            # 'microsoft/Phi-3.5-vision-instruct': {}, #error
            # 'google/gemma-3-4b-it': {}, # bad
            # 'microsoft/OmniParser-v2.0': {}, #error
            # 'U4R/StructTable-base': {}, #StructEqTable. cannot use chat template input
            # 'U4R/StructTable-InternVL2-1B': {}, #StructEqTable. must be used with custom code to correctly load the model
            # 'stepfun-ai/GOT-OCR-2.0-hf': {}, #cannot use chat template input
            ### https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md
            # 'llava-hf/llava-1.5-7b-hf': {}, # bad
            # 'llava-hf/vip-llava-7b-hf': {}, # bad
            # 'llava-hf/llava-v1.6-vicuna-7b-hf': {}, # bad
            # 'llava-hf/llava-v1.6-mistral-7b-hf': {}, # bad
            # 'llava-hf/llava-interleave-qwen-7b-hf': {}, # bad
            # 'llava-hf/llama3-llava-next-8b-hf': {}, # bad
            # 'allenai/olmOCR-7B-0225-preview': {}, #olmOCR(OLMo OCR)
            # 'meta-llama/Llama-3.2-11B-Vision': {}, # cannot use chat template input
            # 'meta-llama/Llama-3.2-11B-Vision-Instruct': {}, # bad
            # 'nvidia/Eagle2.5-8B': {}, # must be used with custom code to correctly load the model
            'ByteDance-Seed/UI-TARS-1.5-7B': {'num_beams': 1}, # good
            # 'allenai/Molmo-7B-D-0924': {}, # must be used with custom code to correctly load the model
        }
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        image = Image.open(path_form57_img)

        generation_config_base = {'max_new_tokens': 4096}
        generation_config_beam_sample = {**generation_config_base, 'do_sample': True, 'temperature': 1, 'top_p': 0.95}
        generation_config_greedy_search = {**generation_config_base, 'do_sample': False, 'num_beams': 1}
        generation_config_additional = dict_model_config[model]
        pipe = pipeline(model=model, device_map=device, model_kwargs={})

        if os.path.exists(path_form57_json):
            with open(path_form57_json, 'r') as f:
                dict_form57 = json.load(f)
        
        else:
            list_response = []
            for _ in range(n_generate):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": PROMPT_DICT_FORM57_TEMP},
                        ],
                    }
                ]
                with utils.Timer(path_form57_json):
                    response = pipe(text=messages, return_full_text=False, generate_kwargs={**generation_config_beam_sample, **generation_config_additional})
                list_response.append(response)

            list_dict_form57_temp = []
            for response in list_response:
                output = response[0]['generated_text']
                dict_form57_temp = parse_json_from_output(output)
                list_dict_form57_temp.append(dict_form57_temp)

            ############ merge transcripts into one
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": PROMPT_DICT_FORM57},
                        {"type": "text", "text": '\n\n'.join(map(lambda x: json.dumps(x, indent=4), list_dict_form57_temp))}
                    ],
                }
            ]
            with utils.Timer(path_form57_json):
                response = pipe(text=messages, return_full_text=False, generate_kwargs=generation_config_greedy_search)
            
            output = response[0]['generated_text']
            dict_form57 = parse_json_from_output(output)
            
            with open(path_form57_json, 'w') as f:
                json.dump(dict_form57, f, indent=4)
        
        
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
    
    return dict_form57, dict_form57_group
