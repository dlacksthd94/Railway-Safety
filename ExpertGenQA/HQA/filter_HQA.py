import json
import itertools
import regex as re
import copy

FOLDER_TASK = 'HQA/'

load_file_path = FOLDER_TASK + "HQA_linked.json"
with open(load_file_path, 'r') as file:
    list_doc = json.load(file)

for doc in list_doc:
    file_name = doc['file']
    print(file_name)
    list_qa = doc['qa']
    for i, qa in enumerate(list_qa):
        print(i)
        print(qa['question'])
    print('---------------'*10)

{doc['file']: [] for doc in list_doc}
dict_qa_general = {
    'RCLQAs0811.pdf': [0, 5, 10, 14],
    'Parts 240and242 FAQs post LERB & OCRB 052213.pdf': [],
    'Car_Availability_Q&A.pdf': [],
    'Rail_Platform_Height_Q&A.pdf': [],
    '240242 FAQs on LERB 060413.pdf': [],
    'FAQsLERBOCRB052213.pdf': [0],
    'Level_Boarding_Alternatives_Q&A.pdf': [0],
    'Parts 240 and 242 FAQs on LERB & OCRB 060413.pdf': [],
    'DLCC QA Wheelchairs and Bus-Rail Service-FINAL.pdf': [0, 2, 3, 4, 5, 6, 7],
    '2022_12 PTC FAQs_final.pdf': [2, 3, 4, 5],
    'Obligations_Public_Entity_Q&A.pdf': [0],
    'fasts1304qsas.pdf': [],
    'FAST Initial Qs and As - Provisions Related to Section 4f (CLEARED TO POST 3.30.16).pdf': [],
    'FAST Act Initial Qs and As on Section 1304- FINAL CLEARED 4.6.16.pdf': [],
    'Section_37-42_Apply_Q&A.pdf': [],
    'ASLRRA QA document 021518.pdf': [1],
    'Existing_Freight_Operations_Q&A.pdf': [],
    'Level_Boarding_Private_Entity_Q&A.pdf': [0],
    'Rail-Improvement-Grant-Conditions -Sec22905-FAQs_033023_PDFa-r1.pdf': [],
    'LRI FAQs 2024_PDFa.pdf': [],
    'Consilidated_ADA_Q&A.pdf': [0, 1, 2, 4, 5],
    'Platform_Alteration_Definition_Q&A.pdf': []
}

dict_abbreviation = {
    'RCO': 'remote control operator (RCO)',
    'PTC': 'positive train control (PTC)',
    'CRISI': 'Consolidated Rail Infrastructure and Safety Improvements Grant Program (CRISI)'
}

dict_qa_topic_needed = {
    'RCLQAs0811.pdf': [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13],
    'Parts 240and242 FAQs post LERB & OCRB 052213.pdf': [0, 1, 2, 3, 4, 5, 6],
    'Car_Availability_Q&A.pdf': [],
    'Rail_Platform_Height_Q&A.pdf': [],
    '240242 FAQs on LERB 060413.pdf': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'FAQsLERBOCRB052213.pdf': [1, 2, 3, 4, 5, 6],
    'Level_Boarding_Alternatives_Q&A.pdf': [],
    'Parts 240 and 242 FAQs on LERB & OCRB 060413.pdf': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'DLCC QA Wheelchairs and Bus-Rail Service-FINAL.pdf': [1],
    '2022_12 PTC FAQs_final.pdf': [],
    'Obligations_Public_Entity_Q&A.pdf': [],
    'fasts1304qsas.pdf': [],
    'FAST Initial Qs and As - Provisions Related to Section 4f (CLEARED TO POST 3.30.16).pdf': [],
    'FAST Act Initial Qs and As on Section 1304- FINAL CLEARED 4.6.16.pdf': [],
    'Section_37-42_Apply_Q&A.pdf': [],
    'ASLRRA QA document 021518.pdf': [],
    'Existing_Freight_Operations_Q&A.pdf': [],
    'Level_Boarding_Private_Entity_Q&A.pdf': [],
    'Rail-Improvement-Grant-Conditions -Sec22905-FAQs_033023_PDFa-r1.pdf': [],
    'LRI FAQs 2024_PDFa.pdf': [],
    'Consilidated_ADA_Q&A.pdf': [],
    'Platform_Alteration_Definition_Q&A.pdf': []
}

assert sum([idx in dict_qa_topic_needed[k] for k, v in dict_qa_general.items() for idx in v]) == 0, "!!!Exception raised!!!\n"*10

### expand abbreviation into full form 
list_doc_abbr_to_full_name = []
for doc in list_doc:
    doc = copy.deepcopy(doc)
    file_name = doc['file']
    list_qa = doc['qa']
    list_qa_abbr_to_full_name = []
    for qa in list_qa:
        pattern = re.compile(rf"(?<!\()(?<![a-zA-Z0-9]){'|'.join(dict_abbreviation.keys())}(?!\))(?![a-zA-Z0-9])")
        result = pattern.sub(lambda x: dict_abbreviation[x.group()], qa['question'])
        qa['question'] = result
        list_qa_abbr_to_full_name.append(qa)
    doc['qa'] = list_qa_abbr_to_full_name
    list_doc_abbr_to_full_name.append(doc)
list_doc_abbr_to_full_name[0]['qa'][0]['question']

### select only general questions
list_doc_filtered = []
for doc in list_doc_abbr_to_full_name:
    doc = copy.deepcopy(doc)
    file_name = doc['file']
    list_qa = doc['qa']
    list_qa_filtered = []
    for idx, qa in enumerate(list_qa):
        if idx in dict_qa_general[file_name]:
            list_qa_filtered.append(qa)
    doc['qa'] = list_qa_filtered
    if list_qa_filtered != []:
        list_doc_filtered.append(doc)

save_file_path = FOLDER_TASK + "HQA_filtered.json"
with open(save_file_path, 'w') as file:
    json.dump(list_doc_filtered, file, indent=4)

### add topic tag to some questions
dict_qa_topic = {
    'RCLQAs0811.pdf': 'remote control operator (RCO)',
    'Parts 240and242 FAQs post LERB & OCRB 052213.pdf': 'administrative process beyond the Locomotive Engineer Review Board and Operating Crew Review Board',
    'Car_Availability_Q&A.pdf': '',
    'Rail_Platform_Height_Q&A.pdf': '',
    '240242 FAQs on LERB 060413.pdf': 'initial stage of FRA\'s dispute resolution process for both the Locomotive Engineer Review Board and Operating Crew Review Board',
    'FAQsLERBOCRB052213.pdf': 'administrative process beyond the Locomotive Engineer Review Board and Operating Crew Review Board',
    'Level_Boarding_Alternatives_Q&A.pdf': '',
    'Parts 240 and 242 FAQs on LERB & OCRB 060413.pdf': 'initial stage of FRA\'s dispute resolution process for both the Locomotive Engineer Review Board and Operating Crew Review Board',
    'DLCC QA Wheelchairs and Bus-Rail Service-FINAL.pdf': 'Wheelchairs and Bus and Rail Service',
    '2022_12 PTC FAQs_final.pdf': '',
    'Obligations_Public_Entity_Q&A.pdf': '',
    'fasts1304qsas.pdf': '',
    'FAST Initial Qs and As - Provisions Related to Section 4f (CLEARED TO POST 3.30.16).pdf': '',
    'FAST Act Initial Qs and As on Section 1304- FINAL CLEARED 4.6.16.pdf': '',
    'Section_37-42_Apply_Q&A.pdf': '',
    'ASLRRA QA document 021518.pdf': '',
    'Existing_Freight_Operations_Q&A.pdf': '',
    'Level_Boarding_Private_Entity_Q&A.pdf': '',
    'Rail-Improvement-Grant-Conditions -Sec22905-FAQs_033023_PDFa-r1.pdf': '',
    'LRI FAQs 2024_PDFa.pdf': '',
    'Consilidated_ADA_Q&A.pdf': '',
    'Platform_Alteration_Definition_Q&A.pdf': ''
}

list_doc_filtered_w_topic = []
for doc in list_doc_abbr_to_full_name:
    doc = copy.deepcopy(doc)
    file_name = doc['file']
    list_qa = doc['qa']
    list_qa_filtered_w_topic = []
    for idx, qa in enumerate(list_qa):
        if idx in dict_qa_topic_needed[file_name]:
            qa['question'] = f'[{dict_qa_topic[file_name]}] ' + qa['question']
            list_qa_filtered_w_topic.append(qa)
        if idx in dict_qa_general[file_name]:
            list_qa_filtered_w_topic.append(qa)
    doc['qa'] = list_qa_filtered_w_topic
    if list_qa_filtered_w_topic != []:
        list_doc_filtered_w_topic.append(doc)

save_file_path = FOLDER_TASK + "HQA_filtered_w_topic.json"
with open(save_file_path, 'w') as file:
    json.dump(list_doc_filtered_w_topic, file, indent=4)
