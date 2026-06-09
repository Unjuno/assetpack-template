#!/usr/bin/env bash
set +e
python -m pip install --upgrade pip
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir huggingface_hub safetensors transformers accelerate diffusers pillow
python scripts/probe_bonsai_tokenizer_smoke.py
python scripts/probe_bonsai_text_embedding_smoke.py
python scripts/probe_bonsai_prompt_context_select.py
bash scripts/ci_prompt_staged.sh
python scripts/validate_bonsai_smoke_artifacts.py
exit 0
