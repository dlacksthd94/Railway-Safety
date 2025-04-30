def inference(tokenizer, model, prompt, config, device):
    input = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    output = model.generate(
        input.input_ids,
        attention_mask=input.attention_mask,
        **config
    )
    input_size = input.input_ids.shape[1]
    idx_first_output_token = input_size
    first_output_token_id = output[:, idx_first_output_token:].squeeze()
    answer = tokenizer.decode(first_output_token_id, skip_special_tokens=False)
    return answer

if __name__ == "__main__":
    pass