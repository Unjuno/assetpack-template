from bonsai_prompt_context import load_prompt_context
from bonsai_prompt_context_select import select_staged_context

def ctx3():
    return select_staged_context(load_prompt_context())
