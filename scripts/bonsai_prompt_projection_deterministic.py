def deterministic_project_prompt_hidden(hidden):
    from bonsai_text_consts import BONSAI_CONTEXT_DIM,TEXT_HIDDEN_DIM
    if hidden.ndim!=3:
        raise ValueError('hidden rank')
    if hidden.shape[-1]!=TEXT_HIDDEN_DIM:
        raise ValueError('hidden dim')
    x=hidden.float()
    y=x.repeat(1,1,BONSAI_CONTEXT_DIM//TEXT_HIDDEN_DIM)
    return y
