#!/usr/bin/env bash
set +e
OUT=reports/bonsai-prompt-adapter-contract
mkdir -p "$OUT"
python scripts/probe_bonsai_prompt_adapter_contract.py
if [ ! -f "$OUT/report.json" ]; then
  echo '{"ok":false,"error":"missing report"}' > "$OUT/report.json"
fi
exit 0
