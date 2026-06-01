#!/usr/bin/env python3
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

src = Path(sys.argv[1])
model = sys.argv[2]
out = Path("generated") / model / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
out.mkdir(parents=True, exist_ok=True)

for name in ["report.json"]:
    p = src / name
    if p.exists():
        shutil.copy2(p, out / name)

for p in src.glob("*.png"):
    shutil.copy2(p, out / p.name)

Path("reports").mkdir(exist_ok=True)
index = Path("reports/model-results.jsonl")
report_path = out / "report.json"
if report_path.exists():
    data = json.loads(report_path.read_text(encoding="utf-8"))
    for row in data.get("results", []):
        row["stored_path"] = str(out)
        index.open("a", encoding="utf-8").write(json.dumps(row, ensure_ascii=False) + "\n")

print(str(out))
