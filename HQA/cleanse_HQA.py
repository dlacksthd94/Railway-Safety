import json
import itertools
import regex as re

FOLDER_TASK = 'HQA/'

load_file_path = FOLDER_TASK + "HQA_extracted.json"
with open(load_file_path, 'r') as file:
    list_doc = json.load(file)

list_remove = [
    '20160726_FAQ_EJOutreach_Final_ENGLISH.pdf',
    'Dallas to Houston DEIS Public Hearings FAQ Sheet.pdf',
    'FAQ.pdf',
]

list_doc_removed = list(filter(lambda x: x['file'] not in list_remove, list_doc))
assert len(list_doc) - len(list_doc_removed) == 3

list_doc_squeezed = []
for doc_removed in list_doc_removed:
    doc_squeezed = {}
    doc_squeezed['file'] = doc_removed['file']
    doc_squeezed['qa'] = list(itertools.chain.from_iterable(doc_removed['qa']))
    list_doc_squeezed.append(doc_squeezed)

list_doc_w_context = []
for doc_squeezed in list_doc_squeezed:
    doc_w_context = {}
    doc_w_context['file'] = doc_squeezed['file']
    # doc_w_context['qa'] = list(filter(lambda qa: qa['context'] not in ['', 'Empty'], doc_squeezed['qa']))
    doc_w_context['qa'] = list(filter(lambda qa: re.search(r'(49 (CFR|C\.F\.R))|(USC|U\.S\.C)', qa['context']), doc_squeezed['qa']))
    list_doc_w_context.append(doc_w_context)
    # for qa in doc_w_context['qa']:
    #     qa['context']

def make_context_list(qa):
    context = qa['context']
    context = context.replace('C.F.R', 'CFR').replace('U.S.C.', 'USC').replace('§', '')
    list_context = re.split(r';|,', context)
    list_context_clean = []
    current_doc = ''
    current_title = ''
    for context in list_context:
        context = context.strip()
        if 'CFR' in context:
            current_doc = 'CFR'
        elif 'USC' in context:
            current_doc = 'USC'
            current_title = re.search(r'\d+(?=\s?USC)', context).group()
        elif re.search('(A|a)pp|(F|f)ed|(R|r)eg|(P|p)art|ADAAG', context):
            current_doc = ''
        context_clean = ''
        if current_doc == 'CFR':
            context = re.sub(r'(\d+)?\s?CFR', '', context)
            idx_text = re.search(r'\d{1,3}\.\d{1,4}(\([a-z]\))?(\(\d+\))?', context)
            if idx_text:
                context_clean = f'49 CFR {idx_text.group()}'
        if current_doc == 'USC':
            context = re.sub(r'(\d+)?\s?USC', '', context)
            idx_text = re.search(r'\d{1,5}(\([a-z]\))?(\(\d+\))?(\([A-Z]\))?', context)
            if idx_text:
                context_clean = f'{current_title} USC {idx_text.group()}'
        if context_clean != '':
            list_context_clean.append(context_clean)
    qa['context'] = list_context_clean
    return qa

list_doc_w_context_list = []
for doc_w_context in list_doc_w_context:
    doc_w_context_list = {}
    doc_w_context_list['file'] = doc_w_context['file']
    doc_w_context_list['qa'] = list(map(make_context_list, doc_w_context['qa']))
    list_doc_w_context_list.append(doc_w_context_list)

list_doc_w_context_dropped = []
for doc_w_context_list in list_doc_w_context_list:
    doc_w_context_dropped = {}
    doc_w_context_dropped['file'] = doc_w_context_list['file']
    doc_w_context_dropped['qa'] = list(filter(lambda qa: qa['context'] != [], doc_w_context_list['qa']))
    if doc_w_context_dropped['qa'] != []:
        list_doc_w_context_dropped.append(doc_w_context_dropped)
    for qa in doc_w_context_dropped['qa']:
        qa['context']

save_file_path = FOLDER_TASK + "HQA_cleansed.json"
with open(save_file_path, 'w') as file:
    json.dump(list_doc_w_context_dropped, file, indent=4)