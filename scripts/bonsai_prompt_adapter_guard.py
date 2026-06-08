from bonsai_prompt_adapter_kind import adapter_kind


def require_repeat_adapter():
    k=adapter_kind()
    if k!='repeat':
        raise ValueError('unsupported adapter kind: '+str(k))
    return k
