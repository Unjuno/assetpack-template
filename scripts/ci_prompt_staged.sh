#!/usr/bin/env bash
set +e
bash scripts/ci_prompt_adapter_contract.sh
bash scripts/ci_prompt_projection_contract.sh
bash scripts/ci_prompt_projection_shape_contract.sh
python scripts/probe_bonsai_prompt_staged_smoke.py
bash scripts/ci_prompt_staged_vae.sh
exit 0
