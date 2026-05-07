import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
EPS = 1e-12


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
        torch_dtype=dtype,
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


def run_token_by_token(model, tokenizer, input_ids, attention_mask, layer_idx, step_fn):
    """
    Run the model token-by-token, calling step_fn(recurrent_state) at each step.
    recurrent_state may be None for non-recurrent layers.
    Returns the list of token texts.
    """
    past_key_values = None
    token_texts = []

    with torch.no_grad():
        for t in range(input_ids.shape[1]):
            # Feed one token at a time, growing the attention mask
            current_ids = input_ids[:, t:t+1]
            current_mask = attention_mask[:, :t+1]

            outputs = model(
                input_ids=current_ids,
                attention_mask=current_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            # Cache carries the recurrent state between steps
            past_key_values = outputs.past_key_values
            token_text = tokenizer.decode(current_ids[0])
            token_texts.append(token_text)

            # Extract recurrent state for the target layer and pass to callback
            recurrent_state = past_key_values.recurrent_states[layer_idx]
            step_fn(recurrent_state)

    return token_texts


# State flattening helpers

def flatten_recurrent_state_to_vector(state_tensor):
    """[1, heads, kdim, vdim] -> flat 1D vector."""
    s = state_tensor[0].detach().float().cpu().numpy()
    if s.ndim != 3:
        raise ValueError(f"Expected 3D state after removing batch dim, got shape {s.shape}")
    return s.reshape(-1)


def flatten_recurrent_state_to_matrix(state_tensor):
    """[1, heads, kdim, vdim] -> [heads*kdim, vdim] 2D matrix."""
    s = state_tensor[0].detach().float().cpu().numpy()
    if s.ndim != 3:
        raise ValueError(f"Expected 3D state after removing batch dim, got shape {s.shape}")
    h, kdim, vdim = s.shape
    return s.reshape(h * kdim, vdim)


# Analysis helpers

def cosine_similarity(a, b, eps=EPS):
    # Returns NaN if either vector is near-zero (direction undefined)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < eps or norm_b < eps:
        return np.nan
    return float(np.dot(a, b) / (norm_a * norm_b + eps))


def numerical_rank_from_svals(svals, tol=1e-5):
    # Count singular values above tol * largest singular value
    if len(svals) == 0 or svals[0] <= 0:
        return 0
    threshold = tol * svals[0]
    return int(np.sum(svals > threshold))


def stable_rank_from_svals(svals):
    # Stable rank = ||A||_F^2 / ||A||_2^2 = sum(svals^2) / max(sval)^2
    if len(svals) == 0 or svals[0] <= 0:
        return 0.0
    return float(np.sum(svals ** 2) / (svals[0] ** 2))


def effective_rank_from_svals(svals, eps=EPS):
    # Effective rank = exp(entropy of normalized singular values), smoother than numerical rank
    total = np.sum(svals)
    if total <= eps:
        return 0.0
    p = svals / total
    p = p[p > eps]
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))
