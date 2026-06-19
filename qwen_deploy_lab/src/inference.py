from model_loader import load_model, load_tokenizer

_model = None
_tokenizer = None

def get_model_and_tokenizer():
    global _model, _tokenizer
    if _model is None:
        _model = load_model()
        _tokenizer = load_tokenizer()
    return _model, _tokenizer

def generate_response(prompt: str, max_tokens: int = 20):
    model, tokenizer = get_model_and_tokenizer()
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False
    )
    # 只取新生成的 token
    new_tokens = outputs[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response