#!/usr/bin/env bash
set +e
OUT=reports/bonsai-prompt-projection-contract
mkdir -p "$OUT"
python scripts/probe_bonsai_prompt_projection_contract.py
if [ ! -f "$OUT/report.json" ]; then
  echo '{"ok":false,"error":"missing report"}' > "$OUT/report.json"
fi
exit 0
