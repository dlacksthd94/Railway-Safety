export CUDA_VISIBLE_DEVICES=0
python main.py --c_api Huggingface --c_model Qwen/Qwen2.5-VL-7B-Instruct --c_n_generate 4 --c_json_source img --r_question_batch group
python main.py --c_api Huggingface --c_model Qwen/Qwen2.5-VL-32B-Instruct --c_n_generate 4 --c_json_source img --r_question_batch group

# python main.py --c_api Huggingface --c_model OpenGVLab/InternVL3-8B-hf --c_n_generate 4 --c_json_source img --r_question_batch group
# python main.py --c_api Huggingface --c_model OpenGVLab/InternVL3-14B-hf --c_n_generate 4 --c_json_source img --r_question_batch group
### python main.py --c_api Huggingface --c_model OpenGVLab/InternVL3-38B-hf --c_n_generate 4 --c_json_source img --r_question_batch group

### python main.py --c_api Huggingface --c_model google/gemma-3-12b-it --c_n_generate 4 --c_json_source img --r_question_batch group
### python main.py --c_api Huggingface --c_model google/gemma-3-27b-it --c_n_generate 4 --c_json_source img --r_question_batch group

### python main.py --c_api Huggingface --c_model llava-hf/llava-1.5-13b-hf --c_n_generate 4 --c_json_source img --r_question_batch group
### python main.py --c_api Huggingface --c_model llava-hf/vip-llava-13b-hf --c_n_generate 4 --c_json_source img --r_question_batch group
### python main.py --c_api Huggingface --c_model llava-hf/llava-v1.6-34b-hf --c_n_generate 4 --c_json_source img --r_question_batch group

### python main.py --c_api Huggingface --c_model ByteDance-Seed/UI-TARS-1.5-7B --c_n_generate 4 --c_json_source img --r_question_batch group

python main.py --c_api None --c_model None --c_n_generate 0 --c_json_source csv --r_question_batch single

python main.py --c_api OpenAI --c_model o4-mini --c_n_generate 4 --c_json_source img --r_question_batch group
