#!/usr/bin/env bash
set +e
python scripts/probe_bonsai_prompt_staged_smoke.py
bash scripts/ci_prompt_staged_vae.sh
exit 0
