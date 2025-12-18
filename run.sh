### export CUDA_VISIBLE_DEVICES=0
### python main.py --c_api Huggingface --c_model Qwen/Qwen2.5-VL-32B-Instruct --c_n_generate 4 --c_json_source img --r_question_batch group
# python main.py --c_api Huggingface --c_model Qwen/Qwen2.5-VL-72B-Instruct --c_n_generate 4 --c_json_source img --r_question_batch all --r_api Huggingface --r_model Qwen/Qwen2.5-VL-72B-Instruct
# python main.py --c_api Huggingface --c_model Qwen/Qwen2.5-VL-72B-Instruct --c_n_generate 4 --c_json_source img --r_question_batch group --r_api Huggingface --r_model microsoft/phi-4
# python main.py --c_api Huggingface --c_model Qwen/Qwen3-VL-32B-Instruct --c_n_generate 4 --c_json_source img --r_question_batch group --r_api Huggingface --r_model microsoft/phi-4

### python main.py --c_api Huggingface --c_model OpenGVLab/InternVL3-38B-hf --c_n_generate 4 --c_json_source img --r_question_batch group --r_api Huggingface --r_model microsoft/phi-4
### python main.py --c_api Huggingface --c_model OpenGVLab/InternVL3-78B-hf --c_n_generate 4 --c_json_source img --r_question_batch group --r_api Huggingface --r_model microsoft/phi-4
# python main.py --c_api Huggingface --c_model OpenGVLab/InternVL3_5-38B-HF --c_n_generate 4 --c_json_source img --r_question_batch group --r_api Huggingface --r_model microsoft/phi-4

### python main.py --c_api Huggingface --c_model ByteDance-Seed/UI-TARS-72B-DPO --c_n_generate 4 --c_json_source img --r_question_batch group --r_api Huggingface --r_model microsoft/phi-4

### python main.py --c_api Huggingface --c_model llava-hf/llava-v1.6-34b-hf --c_n_generate 4 --c_json_source img --r_question_batch group --r_api Huggingface --r_model microsoft/phi-4
### python main.py --c_api Huggingface --c_model llava-hf/llava-next-72b-hf --c_n_generate 4 --c_json_source img --r_question_batch group --r_api Huggingface --r_model microsoft/phi-4
### python main.py --c_api Huggingface --c_model llava-hf/llava-onevision-qwen2-72b-ov-hf --c_n_generate 4 --c_json_source img --r_question_batch group --r_api Huggingface --r_model microsoft/phi-4
### python main.py --c_api Huggingface --c_model meta-llama/Llama-3.2-90B-Vision-Instruct --c_n_generate 4 --c_json_source img --r_question_batch group --r_api Huggingface --r_model microsoft/phi-4

############### main
### CSV source
# python main.py --c_api None --c_model None --c_n_generate 0 --c_json_source csv --r_question_batch single --r_api Huggingface --r_model microsoft/phi-4

for s in 1 2 3 4; do
    for b in single group all; do
        python main.py --c_api Huggingface --c_model Qwen/Qwen3-VL-32B-Instruct --c_n_generate 4 --c_json_source img --c_seed "$s" --r_question_batch "$b" --r_api Huggingface --r_model microsoft/phi-4
        python main.py --c_api Huggingface --c_model OpenGVLab/InternVL3_5-38B-HF --c_n_generate 4 --c_json_source img --c_seed "$s" --r_question_batch "$b" --r_api Huggingface --r_model microsoft/phi-4
        python main.py --c_api OpenAI --c_model o4-mini --c_n_generate 4 --c_json_source img --c_seed "$s" --r_question_batch "$b" --r_api Huggingface --r_model microsoft/phi-4
    done
done

for s in 2; do
    for b in single group all; do
        python main.py --c_api Huggingface --c_model OpenGVLab/InternVL3_5-38B-HF --c_n_generate 4 --c_json_source img --c_seed "$s" --r_question_batch "$b" --r_api Huggingface --r_model microsoft/phi-4
    done
done
