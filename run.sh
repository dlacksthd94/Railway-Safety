# python main.py --c_api Huggingface --c_model Qwen/Qwen2.5-VL-7B-Instruct --c_n_generate 4 --c_json_source img --r_question_batch single
python main.py --c_api Huggingface --c_model Qwen/Qwen2.5-VL-7B-Instruct --c_n_generate 4 --c_json_source img --r_question_batch group
# CUDA_VISIBLE_DEVICES=1 python main.py --c_api Huggingface --c_model OpenGVLab/InternVL3-8B-hf --c_n_generate 4 --c_json_source img --r_question_batch single
CUDA_VISIBLE_DEVICES=1 python main.py --c_api Huggingface --c_model OpenGVLab/InternVL3-8B-hf --c_n_generate 4 --c_json_source img --r_question_batch group
# CUDA_VISIBLE_DEVICES=1 python main.py --c_api Huggingface --c_model ByteDance-Seed/UI-TARS-1.5-7B --c_n_generate 4 --c_json_source img