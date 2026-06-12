#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--allowed-claim", default="ci_image_model_remaining_cat_measurement_not_model_quality_claim")
    args = parser.parse_args()

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            report = {}
    else:
        report = {}

    report.setdefault("experiment_id", "image-model-ci-remaining-cats-v1")
    record = {
        "candidate": {"id": args.candidate_id},
        "status": "failed",
        "error_type": "CandidateTimeout",
        "error": f"Candidate benchmark exceeded {args.timeout_seconds} seconds.",
        "execution_attempted": True,
        "candidate_timeout_seconds": args.timeout_seconds,
        "total_seconds": args.timeout_seconds,
    }

    candidates = report.get("candidates")
    if isinstance(candidates, list) and candidates:
        candidates[-1].update(record)
    else:
        report["candidates"] = [record]

    report["status"] = "failed"
    report["ci_conclusion"] = "success_with_candidate_failures"
    report["claim_promotable_to_manifest"] = False
    report["allowed_claim"] = args.allowed_claim
    report["candidate_timeout_seconds"] = args.timeout_seconds
    report["timeout_recorded_at_unix"] = int(time.time())
    report["summary"] = {"passed": 0, "failed": 1, "skipped": 0, "total": 1}

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
