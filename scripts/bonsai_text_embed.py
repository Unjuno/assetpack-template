import os
import torch
from bonsai_text_consts import PROMPTS, REPO, TOKENIZER_SUBFOLDER, TEXT_ENCODER_SUBFOLDER


def load_text_hidden(prompts=PROMPTS):
    from transformers import AutoModel, AutoTokenizer
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    tok = AutoTokenizer.from_pretrained(REPO, subfolder=TOKENIZER_SUBFOLDER, trust_remote_code=True, token=token)
    enc = tok(prompts, padding=True, truncation=True, return_tensors