from bonsai_text_consts import REPO, TOKENIZER_SUBFOLDER

def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(REPO, subfolder=TOKENIZER_SUBFOLDER, trust_remote_code=True)

def tokenize(prompts, tok):
    return tok(prompts, padding=True, truncation=True, return_tensors='pt')
