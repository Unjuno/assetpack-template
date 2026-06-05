def run_text_smoke():
    from bonsai_text_consts import PROMPTS
    from bonsai_tokenize import load_tokenizer, tokenize
    from bonsai_text_model import load_text_model
    from bonsai_text_forward import text_hidden
    from bonsai_prompt_adapter import adapt_prompt_hidden
    from bonsai_tensor_stats import tensor_stat
    h=text_hidden(load_text_model(), tokenize(PROMPTS, load_tokenizer()))
    c=adapt_prompt_hidden(h)
    return {'hidden':tensor_stat(h),'context':tensor_stat(c)}
