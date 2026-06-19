from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from config import MODEL_PATH, DEVICE

def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto"
    )
    return model

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return tokenizer