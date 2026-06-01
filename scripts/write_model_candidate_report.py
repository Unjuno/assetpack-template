#!/usr/bin/env python3
import json
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def main() -> int:
    if yaml is None:
        raise SystemExit("PyYAML is required")

    cfg = yaml.safe_load(Path("experiments/model-candidates.yml").read_text(encoding="utf-8"))
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)

    rows = []
    for item in cfg.get("candidates", []):
        rows.append({
            "experiment_id": cfg.get("id"),
            "model_id": item.get("id"),
            "enabled": bool(item.get("enabled")),
            "backend": item.get("backend"),
            "model_ref": item.get("model_ref"),
            "role": item.get("role"),
            "runner": cfg.get("runner"),
            "width": cfg.get("limits", {}).get("width"),
            "height": cfg.get("limits", {}).get("height"),
            "status": "planned",
            "result": "not_run_yet"
        })

    Path("reports/model-candidates.latest.json").write_text(
        json.dumps({"experiment": cfg.get("id"), "candidates": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )

    with Path("reports/model-candidates.index.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} candidate rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
