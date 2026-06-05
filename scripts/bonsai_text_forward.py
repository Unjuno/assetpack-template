def text_hidden(model, enc):
    import torch
    with torch.inference_mode():
        out = model(**enc)
    return out.last_hidden_state
