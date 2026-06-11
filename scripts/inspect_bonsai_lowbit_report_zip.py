#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

FORBIDDEN_TRUE_FLAGS = [
    "is_real_transformer_block",
    "is_full_bonsai_pipeline",
]


def load_json_reports(zip_path: Path) -> dict[str, dict[str, Any]]:
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    reports: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if not name.endswith("report.json"):
                continue
            with archive.open(name) as handle:
                data = json.loads(handle.read().decode("utf-8"))
            if not isinstance(data, dict):
                raise AssertionError(f"{name} is not a JSON object")
            reports[name] = data
    return reports


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def find_report(reports: dict[str, dict[str, Any]], report_path: str | None, graph_kind: str | None) -> tuple[str, dict[str, Any]]:
    if report_path:
        require(report_path in reports, f"missing report path {report_path}; available={sorted(reports)}")
        return report_path, reports[report_path]
    if graph_kind:
        matches = [(path, report) for path, report in reports.items() if report.get("graph_kind") == graph_kind]
        require(len(matches) == 1, f"expected exactly one graph_kind={graph_kind}, got {len(matches)}")
        return matches[0]
    raise AssertionError("provide --report-path or --graph-kind")


def validate(args: argparse.Namespace) -> dict[str, Any]:
    reports = load_json_reports(Path(args.zip))
    path, report = find_report(reports, args.report_path, args.graph_kind)

    if args.ok is not None:
        require(report.get("ok") is args.ok, f"expected ok={args.ok}, got {report.get('ok')}")
    if args.graph_kind:
        require(report.get("graph_kind") == args.graph_kind, f"expected graph_kind={args.graph_kind}, got {report.get('graph_kind')}")
    if args.sequence_block_count is not None:
        require(int(report.get("sequence_block_count", -1)) == args.sequence_block_count, f"bad sequence_block_count: {report.get('sequence_block_count')}")
    if args.onnx_segment_count is not None:
        require(int(report.get("onnx_segment_count", -1)) == args.onnx_segment_count, f"bad onnx_segment_count: {report.get('onnx_segment_count')}")
    if args.critical_allclose:
        require(report.get("critical_outputs_allclose_rtol_1e_4_atol_1e_5") is True, "critical outputs are not allclose")
    if args.max_critical_error is not None:
        value = float(report.get("critical_max_abs_error", "inf"))
        require(value <= args.max_critical_error, f"critical_max_abs_error {value} > {args.max_critical_error}")
    if args.not_monolithic:
        require(report.get("is_single_monolithic_onnx") is False, f"expected segmented, got is_single_monolithic_onnx={report.get('is_single_monolithic_onnx')}")
    if args.no_forbidden_claims:
        for flag in FORBIDDEN_TRUE_FLAGS:
            require(report.get(flag) is not True, f"forbidden true flag: {flag}")
        claim = str(report.get("claim", ""))
        forbidden_terms = ["full_bonsai_pipeline_verified", "real_transformer_block_verified", "prompt_to_image"]
        for term in forbidden_terms:
            require(term not in claim, f"forbidden claim term in claim: {term}")

    return {
        "ok": True,
        "zip": str(args.zip),
        "report_path": path,
        "graph_kind": report.get("graph_kind"),
        "block_indices": report.get("block_indices"),
        "segment_block_indices": report.get("segment_block_indices"),
        "sequence_block_count": report.get("sequence_block_count"),
        "onnx_segment_count": report.get("onnx_segment_count"),
        "is_single_monolithic_onnx": report.get("is_single_monolithic_onnx"),
        "critical_outputs_allclose_rtol_1e_4_atol_1e_5": report.get("critical_outputs_allclose_rtol_1e_4_atol_1e_5"),
        "critical_max_abs_error": report.get("critical_max_abs_error"),
        "diagnostic_max_abs_error": report.get("diagnostic_max_abs_error"),
        "claim": report.get("claim"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and validate a downloaded Bonsai low-bit report JSON artifact ZIP.")
    parser.add_argument("--zip", required=True, help="Path to downloaded GitHub Actions artifact ZIP")
    parser.add_argument("--report-path", help="Exact report path inside the ZIP")
    parser.add_argument("--graph-kind", help="Expected graph_kind; used to locate the report if --report-path is omitted")
    parser.add_argument("--ok", action=argparse.BooleanOptionalAction, default=None, help="Require report ok boolean")
    parser.add_argument("--sequence-block-count", type=int)
    parser.add_argument("--onnx-segment-count", type=int)
    parser.add_argument("--critical-allclose", action="store_true")
    parser.add_argument("--max-critical-error", type=float)
    parser.add_argument("--not-monolithic", action="store_true")
    parser.add_argument("--no-forbidden-claims", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        summary = validate(parse_args())
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
