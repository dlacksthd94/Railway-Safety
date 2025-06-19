import json
import os
from tqdm import tqdm
import copy

# API, MODEL = ('Google_DocAI', 'form_parser') #'layout_parser' 'form_parser'
# API, MODEL = ('AWS_Textract', 'textract')
# API, MODEL = ('Azure_FormRecognizer', 'form_recognizer')
API, MODEL = ('OpenAI_GPT', 'o4-mini') #'gpt-4.1-mini', 'gpt-4o-mini' 'o4-mini'
# API, MODEL = ('Huggingface', '') #

N_OUTPUT = 5
DATA_FOLDER = 'data/'
FN_PDF_FORM57 = 'FRA F 6180.57 (Form 57) form only.pdf'
FN_IMG_FORM57 = 'FRA F 6180.57 (Form 57) form only.jpg'
FN_JSON_FORM57 = f'form57_field_def_{API}_{MODEL}.json'
FN_JSON_FORM57_MERGE = f'form57_field_def_merge_{API}_{MODEL}.json'
FN_JSON_FORM57_GROUP = f'form57_field_def_group_{API}_{MODEL}.json'
path_pdf_form57 = DATA_FOLDER + FN_PDF_FORM57
path_img_form57 = DATA_FOLDER + FN_IMG_FORM57
path_json_form57 = DATA_FOLDER + FN_JSON_FORM57
path_json_form57_merge = DATA_FOLDER + FN_JSON_FORM57_MERGE
path_json_form57_group = DATA_FOLDER + FN_JSON_FORM57_GROUP

############# Google Document AI API

if API == 'Google_DocAI':
    from google.cloud import documentai_v1 as documentai

    dict_type_pid = {
        'form_parser': 'b4aa30ed34270c72',
        'layout_parser': '489307cad77319fb'
    }
    
    # 1) Configure client
    project_id = "railway-safety-460721"
    location   = "us"                # e.g. "us", "eu"
    processor_id = dict_type_pid[MODEL]
    client = documentai.DocumentProcessorServiceClient(
        client_options={"api_endpoint": f"{location}-documentai.googleapis.com"}
    )

    # 2) Read your PDF
    with open(path_pdf_form57, "rb") as f:
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
if API == 'AWS_Textract':
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

    result_pdf = analyze_pdf(path_pdf_form57, client, feature_type)
    blocks = result_pdf['Blocks']
    draw_boxes(blocks, path_pdf_form57)
    geometry_popped = [block.pop('Geometry') for block in blocks]
    # result_pdf['Blocks'] = [block for block in blocks if block['BlockType'] != 'WORD']
    {block['BlockType'] for block in blocks}
    with open(path_json_form57, 'w') as f:
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

    # images = convert_from_path(path_pdf_form57)
    # images[0].save(path_img_form57, "JPEG")
    # result_img = analyze_img(path_img_form57, client, feature_type)

    # len(result_pdf['Blocks'])
    # len(result_img['Blocks'])

############# Azure Form Recognizer API
# https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/concept/choose-model-feature?view=doc-intel-4.0.0#pretrained-document-analysis-models
if API == 'Azure_FormRecognizer':
    import os
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

    with open(path_pdf_form57, "rb") as f:
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
if API == 'OpenAI_GPT':
    from openai import OpenAI

    API_key = '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'
    client = OpenAI(api_key=API_key)

    with open(path_pdf_form57, "rb") as f:
        pdf_file = client.files.create(
            file=f,
            purpose="user_data"
        )

    ############ transcribe pdf N times
    if os.path.exists(path_json_form57):
        with open(path_json_form57, 'r') as f:
            list_dict_transcript = json.load(f)

    else:
        prompt = """
        Transcribe each field in JSON format:
        index: {
            "name": field name,
            "choices": {
                index: value
            }
        }
        """

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": prompt},
                        { "type": "file", "file": {"file_id": pdf_file.id} },
                    ]
                },
            ],
            n=N_OUTPUT,
            # seed=0,
            # temperature=0,
        )

        list_dict_transcript = []
        for choice in response.choices:
            output = choice.message.content
            try:
                clean_transcript = output.strip('`')
                dict_transcript = json.loads(clean_transcript)
            except:
                json_start_index = output.index('```')
                json_end_index = output.rindex('```')
                clean_transcript = output[json_start_index:json_end_index].strip('`').replace('json', '', 1)
                dict_transcript = json.loads(clean_transcript)
            list_dict_transcript.append(dict_transcript)

        with open(path_json_form57, 'w') as f:
            json.dump(list_dict_transcript, f, indent=4)

    ############ merge transcripts into one
    if os.path.exists(path_json_form57_merge):
        with open(path_json_form57_merge, 'r') as f:
            dict_transcript_merge = json.load(f)
    else:
        prompt = f"""
        I'll give you:
        1. The original form PDF.
        2. A list of {N_OUTPUT} JSON files, each one an imperfect transcription of that same form.
        Please merge them into a single, accurate transcription.
        """
        
        response = client.responses.create(
            model=MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        { "type": "input_file", "file_id": pdf_file.id },
                        { "type": "input_text", "text": prompt},
                        { "type": "input_text", "text": str(list_dict_transcript)}
                    ]
                }
            ],
        )
        # print(response.output_text)

        output = response.output_text
        try:
            clean_output = output.strip('`')
            dict_transcript_merge = json.loads(clean_output)
        except:
            json_start_index = output.index('```')
            json_end_index = output.rindex('```')
            clean_output = output[json_start_index:json_end_index].strip('`').replace('json', '', 1)
            dict_transcript_merge = json.loads(clean_output)
        with open(path_json_form57_merge, 'w') as f:
            json.dump(dict_transcript_merge, f, indent=4)
    
    ############ categorize the entries
    if os.path.exists(path_json_form57_group):
        with open(path_json_form57_group, 'r') as f:
            dict_transcript_group = json.load(f)
    else:
        prompt = """
        I'll give you:
        1. The original form PDF.
        2. A single, accurate transcription of the PDF.
        Please categorize the entries by relevance, making use of the PDF's layout.
        The format should be:
        {
            group: [entry_index, entry_index, ...],
            group: [entry_index, entry_index, ...],
            ...
        }
        """
        
        response = client.responses.create(
            model=MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        { "type": "input_file", "file_id": pdf_file.id },
                        { "type": "input_text", "text": prompt},
                        { "type": "input_text", "text": str(dict_transcript_merge)}
                    ]
                }
            ],
        )
        print(response.output_text)

        output = response.output_text
        try:
            clean_group = output.strip('`')
            dict_group = json.loads(clean_group)
        except:
            json_start_index = output.index('```')
            json_end_index = output.rindex('```')
            clean_group = output[json_start_index:json_end_index].strip('`').replace('json', '', 1)
            dict_group = json.loads(clean_group)
        
        dict_transcript_group = {group: {idx: dict_output[idx] for idx in indices} for group, indices in dict_group.items()}

        with open(path_json_form57_group, 'w') as f:
            json.dump(dict_transcript_group, f, indent=4)


############# Hugginface VLMs
if API == 'Huggingface':
    
    import torch
    import gc
    from transformers import pipeline
    from PIL import Image

    dict_models = {
        'Qwen/Qwen2.5-VL-7B-Instruct': {'temperature': 1, 'top_p': 0.95},
        'microsoft/GUI-Actor-7B-Qwen2.5-VL': {'temperature': 1, 'top_p': 0.95},
        # 'OpenGVLab/InternVL3-8B': {}, #must be used with custom code to correctly load the model
        # 'OpenGVLab/InternVL3-8B-Instruct': {}, #must be used with custom code to correctly load the model
        'OpenGVLab/InternVL3-8B-hf': {'do_sample': True, 'temperature': 1, 'top_p': 0.95},
        # 'microsoft/Phi-3.5-vision-instruct': {}, #error
        'google/gemma-3-4b-it': {},
        # 'microsoft/OmniParser-v2.0': {}, #error
        # 'U4R/StructTable-base': {}, #StructEqTable. cannot use chat template input
        # 'U4R/StructTable-InternVL2-1B': {}, #StructEqTable. must be used with custom code to correctly load the model
        # 'stepfun-ai/GOT-OCR-2.0-hf': {}, #cannot use chat template input
        'llava-hf/llava-v1.6-vicuna-7b-hf': {'do_sample': True, 'temperature': 1, 'top_p': 0.95}, #LLaVA 1.6 https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md
        'llava-hf/llava-v1.6-mistral-7b-hf': {'do_sample': True, 'temperature': 1, 'top_p': 0.95}, #LLaVA 1.6
        'llava-hf/llama3-llava-next-8b-hf': {'do_sample': True, 'temperature': 1, 'top_p': 0.95}, #LLaVA-NeXT
        # 'allenai/olmOCR-7B-0225-preview': {}, #olmOCR(OLMo OCR)
        'meta-llama/Llama-3.2-11B-Vision': {'do_sample': True, 'temperature': 1, 'top_p': 0.95},
        'meta-llama/Llama-3.2-11B-Vision-Instruct': {},
    }
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    image = Image.open(path_img_form57)

    for model_path, config in tqdm(dict_models.items()):
        print('--------------------------------')
        print(model_path)
        pipe = pipeline(model=model_path, device_map=device, model_kwargs={'num_return_sequences': 2})
        print(pipe.generation_config.get_generation_mode())
        print(pipe.generation_config.top_p)
        print(pipe.generation_config.top_k)
        print(pipe.generation_config.temperature)

        # del pipe
        # gc.collect()
        # torch.cuda.empty_cache()
        # continue

        prompt = """
        Transcribe each field in JSON format:
        index: {
            "name": field name,
            "type": "string" OR "digit" OR “single choice”,
            "choices": {
                index: value
            }
        }
        """
        # prompt = "Is this image a report form? Answer in YES or NO."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        response = pipe(text=messages, return_full_text=False, max_new_tokens=409, generate_kwargs=config)
        len(response)
        print(list_response[0][0]['generated_text'])
        list_response[0][0]['generated_text'] == list_response[1][0]['generated_text']
        list_response[0][0]['generated_text'] == list_response[0][1]['generated_text']

        del pipe
        gc.collect()
        torch.cuda.empty_cache()


############# Convert the json to questions

with open(path_json_form57_merge, 'r') as f:
    dict_output = json.load(f)
