#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def load_base_module():
    module_path = Path(__file__).with_name("run_bonsai_transformer_load_probe.py")
    spec = importlib.util.spec_from_file_location("bonsai_transformer_load_probe_base", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base probe module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rewrite_allowed_claims(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: rewrite_allowed_claims(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite_allowed_claims(item) for item in value]
    if value == "bonsai_transformer_minimal_onnxruntime_cpu_load_with_non_float_trig_cast_patch_verified_not_execution":
        return "bonsai_transformer_minimal_onnxruntime_cpu_load_with_double_trig_cast_patch_verified_not_execution"
    return value


def postprocess_report(report_path: Path) -> None:
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report = rewrite_allowed_claims(report)
    report["experiment_id"] = "bonsai-transformer-load-probe-v9"
    report["double_trig_cast_probe_revision"] = "v1"
    report["double_trig_cast_probe_note"] = (
        "This wrapper reuses run_bonsai_transformer_load_probe.py with DOUBLE excluded from "
        "the native ORT CPU trig type set, so DOUBLE Cos/Sin inputs are cast to FLOAT for "
        "load-only patch attempts."
    )
    diagnostics = report.get("onnx_graph_diagnostics")
    if isinstance(diagnostics, dict):
        diagnostics["unsupported_trig_input_type_policy"] = "FLOAT/FLOAT16/BFLOAT16 treated as native; DOUBLE and non-floating types are patch candidates."
    patch = report.get("onnxruntime_trig_cast_patch")
    if isinstance(patch, dict):
        patch["patch_scope"] = "double_and_non_native_trig_inputs"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/bonsai-onnx-smoke.yml")
    parser.add_argument("--out-dir", default="reports/bonsai-transformer-load-probe")
    args = parser.parse_args()

    base = load_base_module()
    # TensorProto: FLOAT=1, FLOAT16=10, BFLOAT16=16. Exclude DOUBLE=11 so Bonsai
    # DOUBLE Cos/Sin nodes are patched for ORT CPU load-only probing.
    base.FLOAT_ELEM_TYPES = {1, 10, 16}
    exit_code = int(base.run(args.config, args.out_dir) or 0)
    postprocess_report(Path(args.out_dir) / "report.json")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
