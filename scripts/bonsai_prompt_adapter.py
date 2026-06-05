def adapt_prompt_hidden(hidden):
    from bonsai_text_consts import BONSAI_CONTEXT_DIM
    factor = BONSAI_CONTEXT_DIM // hidden.shape[-1]
    return hidden.repeat_interleave(factor, dim=-1)
