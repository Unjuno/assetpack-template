#!/usr/bin/env bash
set +e
OUT_DIR="reports/bonsai-prompt-staged-vae-smoke"
mkdir -p "$OUT_DIR"
python scripts/probe_bonsai_prompt_staged_vae_smoke.py
rc=$?
if [ ! -f "$OUT_DIR/report.json" ]; then
  printf '{\n  "target": "prompt context staged transformer native packed unpatchify VAE decode smoke",\n  "ok": false,\n  "decode_success": false,\n  "error": "probe exited without report",\n  "exit_code": %s\n}\n' "$rc" > "$OUT_DIR/report.json"
fi
exit 0
