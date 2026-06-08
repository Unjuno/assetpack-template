from bonsai_prompt_adapter_kind import adapter_name


def prompt_meta():
    return {'adapter':adapter_name(),'loop_mode':'per_step_repredict','claim':'ci_smoke_not_full_prompt_conditioning'}
