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
SOURCE_DIR = ROOT / "bonsai-lowbit-ten-by-two-persistent-onnx-artifacts"
SOURCE = SOURCE_DIR / "report.json"
OUT = ROOT / "bonsai-lowbit-ten-by-two-persistent-onnx-artifacts-validation" / "report.json"
STRICT_RTOL = 1e-4
STRICT_ATOL = 1e-5
BLOCK_INDICES = list(range(20))
SEGMENTS = [[index, index + 1] for index in range(0, 20, 2)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AssertionError(f"missing report: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AssertionError(f"report is not an object: {path}")
    return data


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_file_from_manifest(item: dict[str, Any]) -> Path:
    rel_path = item.get("path")
    require(isinstance(rel_path, str) and rel_path, f"bad segment path: {item}")
    path = SOURCE_DIR / rel_path
    require(path.is_file(), f"missing persisted ONNX segment: {path}")
    expected_sha = item.get("sha256")
    require(sha256_file(path) == expected_sha, f"sha256 mismatch for {path}")
    require(path.stat().st_size == int(item.get("size_bytes", -1)), f"size mismatch for {path}")
    return path


def main() -> None:
    source = load_json(SOURCE)
    require(source.get("artifact_kind") == "persistent_onnx_segment_bundle", f"bad artifact kind: {source}")
    require(source.get("persistent_onnx_artifacts") is True, f"artifact must persist ONNX files: {source}")
    require(source.get("reusable_onnx_chain_artifact") is True, f"artifact must be reusable: {source}")
    require(source.get("onnx_paths_persisted_in_reports") is True, f"ONNX paths must be persisted: {source}")
    require(source.get("uses_lowbit_source") is True, f"export probe did not use lowbit source: {source}")
    require(source.get("writes_expanded_checkpoint") is False, f"export probe wrote expanded checkpoint: {source}")
    require(source.get("constant_folding_disabled") is True, f"constant folding not disabled: {source}")
    require(source.get("unpack_lowering") == "arithmetic_floor_div_mod_no_bitshift", f"unexpected unpack lowering: {source}")
    require(source.get("graph_kind") == "ten_by_two_single_blocks_modulated_attention_to_out_residual_stack", f"unexpected graph kind: {source}")
    require(source.get("block_indices") == BLOCK_INDICES, f"bad block indices: {source}")
    require(source.get("segment_block_indices") == SEGMENTS, f"bad segment block indices: {source}")
    require(int(source.get("sequence_block_count", 0)) == 20, f"bad sequence block count: {source}")
    require(int(source.get("onnx_segment_count", 0)) == 10, f"bad ONNX segment count: {source}")
    require(source.get("is_single_monolithic_onnx") is False, f"must not be monolithic ONNX: {source}")
    require(source.get("is_real_transformer_block") is False, f"must not claim real transformer block: {source}")
    require(source.get("is_full_bonsai_pipeline") is False, f"must not claim full bonsai pipeline: {source}")
    require(source.get("critical_outputs_allclose_rtol_1e_4_atol_1e_5") is True, f"export critical outputs failed: {source}")

    segment_files = source.get("onnx_segment_files", [])
    require(isinstance(segment_files, list) and len(segment_files) == 10, f"expected ten ONNX segment files: {source}")

    reference_rel = source.get("reference_path")
    require(isinstance(reference_rel, str) and reference_rel, f"missing reference path: {source}")
    reference_path = SOURCE_DIR / reference_rel
    require(reference_path.is_file(), f"missing reference arrays: {reference_path}")
    require(sha256_file(reference_path) == source.get("reference_sha256"), f"reference sha256 mismatch: {reference_path}")

    reference = np.load(reference_path)
    hidden = reference["initial_hidden"].astype(np.float32)
    temb = reference["temb"].astype(np.float32)

    validation_items: list[dict[str, Any]] = []
    for expected_segment, item in zip(SEGMENTS, segment_files):
        require(item.get("segment_block_indices") == expected_segment, f"bad segment ordering: {item}")
        segment_path = require_file_from_manifest(item)
        session = ort.InferenceSession(str(segment_path), providers=["CPUExecutionProvider"])
        outputs = session.run(None, {"hidden": hidden, "temb": temb})
        outputs = [out.astype(np.float32) for out in outputs]

        prefix = f"segment{expected_segment[0]}_{expected_segment[1]}"
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
        "name": "chained_final_block19_output_from_persisted_onnx_artifacts",
        "category": "critical",
        "output_shape": list(expected_final.shape),
        "mean_abs_error": float(np.abs(final_diff).mean()),
        "max_abs_error": float(np.abs(final_diff).max()),
        "allclose_rtol_1e_4_atol_1e_5": bool(np.allclose(hidden, expected_final, rtol=STRICT_RTOL, atol=STRICT_ATOL)),
    })

    critical_allclose = all(item["allclose_rtol_1e_4_atol_1e_5"] for item in validation_items)
    critical_max_abs_error = max((item["max_abs_error"] for item in validation_items), default=None)
    total_size = sum((SOURCE_DIR / item["path"]).stat().st_size for item in segment_files)

    summary = {
        "ok": bool(critical_allclose),
        "artifact_kind": "persistent_onnx_segment_bundle_validation",
        "source_report_path": str(SOURCE),
        "graph_kind": source.get("graph_kind"),
        "block_indices": source.get("block_indices"),
        "segment_block_indices": source.get("segment_block_indices"),
        "sequence_block_count": source.get("sequence_block_count"),
        "onnx_segment_count": source.get("onnx_segment_count"),
        "persistent_onnx_artifacts": True,
        "reusable_onnx_chain_artifact": True,
        "validated_without_lowbit_source_reload": True,
        "validated_from_persisted_onnx_files": True,
        "is_single_monolithic_onnx": source.get("is_single_monolithic_onnx"),
        "is_real_transformer_block": source.get("is_real_transformer_block"),
        "is_full_bonsai_pipeline": source.get("is_full_bonsai_pipeline"),
        "critical_outputs_allclose_rtol_1e_4_atol_1e_5": critical_allclose,
        "critical_max_abs_error": critical_max_abs_error,
        "total_onnx_segment_size_bytes": total_size,
        "onnx_segment_files": [
            {
                "segment_block_indices": item.get("segment_block_indices"),
                "path": item.get("path"),
                "sha256": item.get("sha256"),
                "size_bytes": item.get("size_bytes"),
            }
            for item in segment_files
        ],
        "outputs": validation_items,
        "claim": "ten_by_two_persistent_onnx_segments_reusable_chain_artifact_validated_without_lowbit_source_reload_not_single_monolithic_onnx_not_real_transformer_block_or_full_bonsai_pipeline",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not critical_allclose:
        raise AssertionError("persisted ONNX artifact chain critical outputs failed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"ok": False, "error": str(exc)}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        sys.exit(1)
