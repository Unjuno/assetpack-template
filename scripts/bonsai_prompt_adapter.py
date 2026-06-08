def _check(hidden):
    from bonsai_text_consts import BONSAI_CONTEXT_DIM, TEXT_HIDDEN_DIM
    if hidden.ndim != 3:
        raise ValueError('hidden rank')
    if hidden.shape[-1] != TEXT_HIDDEN_DIM:
        raise ValueError('hidden dim')
    if BONSAI_CONTEXT_DIM % TEXT_HIDDEN_DIM != 0:
        raise ValueError('adapter dims')


def _repeat(hidden):
    from bonsai_text_consts import BONSAI_CONTEXT_DIM, TEXT