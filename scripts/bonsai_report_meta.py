from bonsai_prompt_meta import prompt_meta


def add_meta(r):
    r.update(prompt_meta())
    return r
