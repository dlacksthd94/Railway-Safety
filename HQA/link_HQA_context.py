import json
import itertools
import regex as re

FOLDER_TASK = 'HQA/'

load_file_path = FOLDER_TASK + "HQA_cleansed.json"
with open(load_file_path, 'r') as file:
    list_doc = json.load(file)

load_file_path = FOLDER_TASK + "CFR.json"
with open(load_file_path, 'r') as file:
    dict_CFR = json.load(file)

load_file_path = FOLDER_TASK + "USC.json"
with open(load_file_path, 'r') as file:
    dict_USC = json.load(file)

list_doc_linked = []
for doc in list_doc:
    dict_doc = {}
    dict_doc['file'] = doc['file']
    list_qa = []
    for qa in doc['qa']:
        context_concat = ''
        for context in qa['context']:
            if 'CFR' in context:
                idx_subpart = re.search(r'\d{1,3}\.\d{1,4}', context)
                idx_subpart = f'§ {idx_subpart.group()}'
                idx_alphabet = re.search(r'(?<=\d{1,3}\.\d{1,4})\(\w{1,5}\)', context)
                try:
                    dict_subpart = dict_CFR[idx_subpart]
                    if idx_alphabet:
                        context_concat += idx_subpart + idx_alphabet.group() + '\n' + dict_subpart[idx_alphabet.group()]
                    else:
                        context_concat += idx_subpart + '\n' + dict_subpart['all']
                except:
                    pass
            elif 'USC' in context:
                idx_title = re.search(r'\d+(?=\sUSC)', context).group()
                idx_part = re.search(r'(?<=USC\s)\d{1,5}', context).group()
                idx_alphabet = re.search(r'(?<=\d{1,5})\(\w{1,5}\)', context)
                try:
                    part = f'{idx_title} U.S. Code § {idx_part}'
                    dict_part = dict_USC[part]
                    if idx_alphabet:
                        context_concat += part + idx_alphabet.group() + '\n' + dict_part[idx_alphabet.group()]
                    else:
                        context_concat += part + '\n' + dict_part['all']
                except:
                    pass
            context_concat += '\n\n'
        qa['context'] = context_concat
        if qa['context'].strip() != '':
            list_qa.append(qa)
    dict_doc['qa'] = list_qa
    if dict_doc['qa'] != []:
        list_doc_linked.append(dict_doc)

save_file_path = FOLDER_TASK + "HQA_linked.json"
with open(save_file_path, 'w') as file:
    json.dump(list_doc_linked, file, indent=4)