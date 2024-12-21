import json
import itertools
import regex as re

FOLDER_TASK = 'HQA/'

### CFR
load_file_path = FOLDER_TASK + "CFR.txt"
with open(load_file_path, 'r') as file:
    s = file.read()

# list_part = re.split(r'(PART \d+)', s)
# list_part = [list_part[i] + list_part[i + 1] for i in range(1, len(list_part), 2)]

list_subpart = re.split(r'(?<=\n\n)(§ \d+\.\d+)', s)
dict_subpart = {list_subpart[i]: list_subpart[i + 1] for i in range(1, len(list_subpart), 2)}
for idx, text in list(dict_subpart.items()):
    text = text.strip()
    pattern = r"(?<!:\n\n)(?<!;\n\n)(?<=\n\n)(\([a-z]{1}\)|\([a-d]{2}\))\s(.+?)(?=\n\n\([a-z]{1,2}\)|$)"
    list_match = re.findall(pattern, text, flags=re.DOTALL)
    dict_temp = {}
    dict_temp['all'] = text
    for match in list_match:
        dict_temp[match[0]] = match[1].strip()
    dict_subpart[idx] = dict_temp

save_file_path = FOLDER_TASK + "CFR.json"
with open(save_file_path, 'w') as file:
    json.dump(dict_subpart, file, indent=4)

### USC
load_file_path = FOLDER_TASK + "USC.txt"
with open(load_file_path, 'r') as file:
    s = file.read()

list_part = re.split(r'(\d{1,3} .+ § \d+)', s)
dict_part = {list_part[i]: list_part[i + 1] for i in range(1, len(list_part), 2)}
for idx, text in dict_part.items():
    text = text.strip()
    pattern = r"(\([a-z]{1}\)|\([a-f]{2}\))(.+?)(?=\n\([a-z]{1}\))"
    list_match = re.findall(pattern, text, flags=re.DOTALL)
    dict_temp = {}
    dict_temp['all'] = text
    for match in list_match:
        dict_temp[match[0]] = match[1].strip()
    dict_part[idx] = dict_temp

save_file_path = FOLDER_TASK + "USC.json"
with open(save_file_path, 'w') as file:
    json.dump(dict_part, file, indent=4)
