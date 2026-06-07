def adapt_prompt_hidden(hidden):
    from bonsai_text_consts import BONSAI_CONTEXT_DIM, TEXT_HIDDEN_DIM
    if hidden.ndim != 3:
        raise ValueError(f'hidden rank must be 3, got {hidden.ndim}')
    if hidden.shape[-1] != TEXT_HIDDEN_DIM:
        raise ValueError(f'hidden dim must be {TEXT_HIDDEN_DIM}, got {hidden.shape[-1]}')
    if BONSAI_CONTEXT_DIM % TEXT