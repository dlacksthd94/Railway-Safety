import os
import time
from transformers import pipeline
import json
import ast
import shutil

def make_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def remove_dir(path: str) -> None:
    shutil.rmtree(path)

def sanitize_model_path(model_path: str) -> str:
    """Replace path separators so model_path names are safe in folder names."""
    return model_path.replace("/", "--")

def desanitize_model_path(model_path: str) -> str:
    """Replace path separators so model_path names are safe in folder names."""
    return model_path.replace("--", "/")

def generate_openai(client, model_path, content, generation_config=None):
    messages = [
        {
            "role": "user",
            "content": content,
        },
    ]
    response = client.responses.create(model=model_path, input=messages)
    output = response.output_text
    return output

def generate_hf(pipe, model_path, content, generation_config={}):
    messages = [
        {
            "role": "user",
            "content": content,
        },
    ]
    # response = pipe(messages, return_full_text=False, generate_kwargs=generation_config)
    response = pipe(messages, return_full_text=False, **generation_config)
    output = response[0]['generated_text']
    return output

def select_generate_func(api):
    if api == 'OpenAI':
        generate_func = generate_openai
    elif api == 'Huggingface':
        generate_func = generate_hf
    else:
        raise ValueError(f"Unsupported API: {api}")
    return generate_func

class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self._start = time.perf_counter()
        return self           # (optional) so you can read .elapsed later
    
    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self._start
        elapsed = self.format_hms(self.elapsed)
        print(f"[{self.label}]\t elapsed: {elapsed}")
            
    def format_hms(self, total_seconds):
        h = int(total_seconds // 3600)
        m = int((total_seconds % 3600) // 60)
        s = total_seconds % 60               # still a float now
        return f"{h:02d}:{m:02d}:{s:06.3f}"   # e.g. 00:01:02.357

def as_float(val):
    try:
        return float(val)
    except:
        return val

def lower_str(val):
    if isinstance(val, str):
        return val.lower()
    else:
        return val

def parse_json_from_output(output):
    """Parse JSON from the output text of OpenAI response.
    If the output is a plain JSON string, it parses that directly.
    Otherwise, find the code block containing JSON and parse it.
    """
    try:
        if '```' in output:
            json_start_index = output.index('```')
            json_end_index = output.rindex('```')
            str_form57 = output[json_start_index:json_end_index].strip('`')
            if str_form57.startswith('json'):
                str_form57 = str_form57.replace('json', '', 1)
        else:
            str_form57 = output
        try:
            dict_form57 = json.loads(str_form57)
        except:
            dict_form57 = ast.literal_eval(str_form57)
    except:
        dict_form57 = {}
    return dict_form57

def text_binary_classification(pipe, prompt, dict_answer_choice, num_sim):
    list_output = pipe(prompt, max_new_tokens=1, num_return_sequences=num_sim, return_full_text=False)
    list_answer = list(map(lambda output: output['generated_text'].upper(), list_output))
    list_answer_filter = list(filter(lambda answer: answer in dict_answer_choice, list_answer))
    list_answer_map = list(map(lambda answer: dict_answer_choice[answer], list_answer_filter))
    return list_answer_map

def text_generation(pipe, prompt, max_new_tokens=4096):
    output = pipe(prompt, max_new_tokens=max_new_tokens, return_full_text=False)
    answer = output[0]['generated_text']
    return answer
