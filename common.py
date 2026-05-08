import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3.5-0.8B"


def setup_model(model_name=MODEL_NAME):
    # Auto-detect best available device (MPS for Apple Silicon, CUDA for GPU, else CPU)
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Use float32 on CPU/MPS since bfloat16 isn't well supported there
    dtype = torch.float32 if device in ["cpu", "mps"] else torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
    ).to(device)
    model.eval()

    return model, tokenizer, device


def prompt_name(prompt_file):
    # Extract stem (filename without extension) from prompt file path
    return os.path.splitext(os.path.basename(prompt_file))[0]


def tokenize_text(text, tokenizer, device):
    # Tokenize a raw string and return tensors on the target device
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    return input_ids, attention_mask


def tokenize_prompt(prompt_file, tokenizer, device):
    # Read raw text, wrap with the instruct chat template, and tokenize
    with open(prompt_file, "r") as f:
        prompt = f.read()
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenize_text(formatted, tokenizer, device)
