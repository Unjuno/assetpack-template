def load_prompt_context():
    from bonsai_text_consts import PROMPTS
    from bonsai_tokenize import load_tokenizer, tokenize
    from bonsai_text_model import load_text_model
    from bonsai_text_forward import text_hidden
    from bonsai_prompt_adapter import adapt_prompt_hidden
    h=text_hidden(load_text_model(), tokenize(PROMPTS, load_tokenizer()))
    return adapt_prompt_hidden(h)
