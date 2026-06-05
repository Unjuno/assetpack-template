def load_text_model():
    from transformers import AutoModel
    from bonsai_text_consts import REPO, TEXT_ENCODER_SUBFOLDER
    return AutoModel.from_pretrained(REPO, subfolder=TEXT_ENCODER_SUBFOLDER).eval()
