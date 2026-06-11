#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

ROOT = Path("reports")
SEGMENT_ROOT = Path("downloaded-persistent-segments")
REFERENCE_ROOT = Path("downloaded-persistent-reference")
OUT = ROOT / "bonsai-lowbit-ten-by-two-split-persistent-onnx-artifacts-validation" / "report.json"
STRICT_RTOL = 1e-4
STRICT_ATOL = 1e-5
SEGMENTS = [[index, index + 1] for index in range(0, 20, 2)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing json: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(data, dict), f"json is not object: {path}")
    return data


def find_single(pattern: str, root: Path) -> Path:
    matches = sorted(root.glob(pattern))
    require(len(matches) == 1, f"expected one match for {pattern} under {root}, got {[str(m) for m in matches]}")
    return matches[0]


def main() -> None:
    ref_report = load_json(find_single("**/report.json", REFERENCE_ROOT))
    require(ref_report.get("artifact_kind") == "persistent_onnx_chain_reference", f"bad reference report: {ref_report}")
    require(ref_report.get("sequence_block_count") == 20, f"bad reference sequence count: {ref_report}")
    require(ref_report.get("onnx_segment_count") == 10, f"bad reference segment count: {ref_report}")
    ref_path = find_single("**/reference.npz", REFERENCE_ROOT)
    require(sha256_file(ref_path) == ref_report.get("reference_sha256"), f"reference sha256 mismatch: {ref_path}")
    reference = np.load(ref_path)
    hidden = reference["initial_hidden"].astype(np.float32)
    temb = reference["temb"].astype(np.float32)

    segment_reports = []
    for segment in SEGMENTS:
        report = load_json(find_single(f"**/bonsai-lowbit-persistent-onnx-segment-{segment[0]}-{segment[1]}/report.json", SEGMENT_ROOT))
        require(report.get("artifact_kind") == "persistent_onnx_segment", f"bad segment report: {report}")
        require(report.get("block_indices") == segment, f"bad segment block indices: {report}")
        require(report.get("persistent_onnx_artifacts") is True, f"segment not persistent: {report}")
        require(report.get("reusable_onnx_segment_artifact") is True, f"segment not reusable: {report}")
        require(report.get("critical_outputs_allclose_rtol_1e_4_atol_1e_5") is True, f"segment export failed critical: {report}")
        onnx_name = report.get("onnx_path")
        require(isinstance(onnx_name, str) and onnx_name.endswith(".onnx"), f"bad onnx_path: {report}")
        onnx_path = report_path = find_single(f"**/bonsai-lowbit-persistent-onnx-segment-{segment[0]}-{segment[1]}/{onnx_name}", SEGMENT_ROOT)
        require(sha256_file(onnx_path) == report.get("onnx_sha256"), f"segment sha mismatch: {onnx_path}")
        require(onnx_path.stat().st_size == int(report.get("onnx_size_bytes", -1)), f"segment size mismatch: {onnx_path}")
        segment_reports.append((segment, report, onnx_path))

    validation_items: list[dict[str, Any]] = []
    for segment, report, onnx_path in segment_reports:
        prefix = f"segment{segment[0]}_{segment[1]}"
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        outputs = [out.astype(np.float32) for out in session.run(None, {"hidden": hidden, "temb": temb})]
        expected_block0 = reference[f"{prefix}_block0_output"].astype(np.float32)
        expected_block1 = reference[f"{prefix}_block1_output"].astype(np.float32)
        for name, got, expected in [
            (f"{prefix}_block0_output", outputs[0], expected_block0),
            (f"{prefix}_block1_output", outputs[1], expected_block1),
        ]:
            diff = got - expected
            validation_items.append({
                "name": name,
                "category": "critical",
                "output_shape": list(expected.shape),
                "mean_abs_error": float(np.abs(diff).mean()),
                "max_abs_error": float(np.abs(diff).max()),
                "allclose_rtol_1e_4_atol_1e_5": bool(np.allclose(got, expected, rtol=STRICT_RTOL, atol=STRICT_ATOL)),
            })
        hidden = outputs[1]

    expected_final = reference["expected_final_block19_output"].astype(np.float32)
    final_diff = hidden - expected_final
    validation_items.append({
        "name": "chained_final_block19_output_from_split_persisted_onnx_artifacts",
        "category": "critical",
        "output_shape": list(expected_final.shape),
        "mean_abs_error": float(np.abs(final_diff).mean()),
        "max_abs_error": float(np.abs(final_diff).max()),
        "allclose_rtol_1e_4_atol_1e_5": bool(np.allclose(hidden, expected_final, rtol=STRICT_RTOL, atol=STRICT_ATOL)),
    })

    critical_allclose = all(item["allclose_rtol_1e_4_atol_1e_5"] for item in validation_items)
    critical_max_abs_error = max((item["max_abs_error"] for item in validation_items), default=None)
    total_size = sum(path.stat().st_size for _, _, path in segment_reports)

    summary = {
        "ok": bool(critical_allclose),
        "artifact_kind": "split_persistent_onnx_segment_bundle_validation",
        "graph_kind": "ten_by_two_single_blocks_modulated_attention_to_out_residual_stack",
        "block_indices": list(range(20)),
        "segment_block_indices": SEGMENTS,
        "sequence_block_count": 20,
        "onnx_segment_count": 10,
        "persistent_onnx_artifacts": True,
        "reusable_onnx_chain_artifact": True,
        "validated_without_lowbit_source_reload": True,
        "validated_from_persisted_onnx_files": True,
        "is_single_monolithic_onnx": False,
        "is_real_transformer_block": False,
        "is_full_bonsai_pipeline": False,
        "critical_outputs_allclose_rtol_1e_4_atol_1e_5": critical_allclose,
        "critical_max_abs_error": critical_max_abs_error,
        "total_onnx_segment_size_bytes": total_size,
        "onnx_segment_files": [
            {
                "segment_block_indices": segment,
                "artifact_report_sha256": sha256_file(find_single(f"**/bonsai-lowbit-persistent-onnx-segment-{segment[0]}-{segment[1]}/report.json", SEGMENT_ROOT)),
                "onnx_sha256": report.get("onnx_sha256"),
                "onnx_size_bytes": report.get("onnx_size_bytes"),
            }
            for segment, report, _ in segment_reports
        ],
        "reference_sha256": ref_report.get("reference_sha256"),
        "outputs": validation_items,
        "claim": "ten_by_two_split_persistent_onnx_segments_reusable_chain_artifact_validated_without_lowbit_source_reload_not_single_monolithic_onnx_not_real_transformer_block_or_full_bonsai_pipeline",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not critical_allclose:
        raise AssertionError("split persisted ONNX artifact chain critical outputs failed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"ok": False, "error": str(exc)}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        sys.exit(1)
