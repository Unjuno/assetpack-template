#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("reports")
SEGMENT_ROOT = Path("downloaded-persistent-segments")
REFERENCE_ROOT = Path("downloaded-persistent-reference")
VALIDATION_REPORT = ROOT / "bonsai-lowbit-ten-by-two-split-persistent-onnx-artifacts-validation" / "report.json"
OUT = ROOT / "bonsai-lowbit-ten-by-two-chain-handoff" / "report.json"
SEGMENTS = [[index, index + 1] for index in range(0, 20, 2)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array.astype(np.float32, copy=False))
    return hashlib.sha256(arr.tobytes()).hexdigest()


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
    validation = load_json(VALIDATION_REPORT)
    require(validation.get("ok") is True, f"validation report is not ok: {validation}")
    require(validation.get("persistent_onnx_artifacts") is True, f"validation did not use persistent artifacts: {validation}")
    require(validation.get("reusable_onnx_chain_artifact") is True, f"validation did not verify reusable chain: {validation}")
    require(validation.get("validated_without_lowbit_source_reload") is True, f"validation reloaded low-bit source: {validation}")
    require(validation.get("validated_from_persisted_onnx_files") is True, f"validation did not use persisted ONNX files: {validation}")

    ref_report = load_json(find_single("**/report.json", REFERENCE_ROOT))
    ref_path = find_single("**/reference.npz", REFERENCE_ROOT)
    require(sha256_file(ref_path) == ref_report.get("reference_sha256"), f"reference sha256 mismatch: {ref_path}")
    reference = np.load(ref_path)

    handoffs: list[dict[str, Any]] = []
    previous_output_name = "initial_hidden"
    previous_block = None
    for segment in SEGMENTS:
        prefix = f"segment{segment[0]}_{segment[1]}"
        input_name = "initial_hidden" if segment[0] == 0 else previous_output_name
        output0_name = f"{prefix}_block0_output"
        output1_name = f"{prefix}_block1_output"
        input_tensor = reference[input_name].astype(np.float32)
        output0_tensor = reference[output0_name].astype(np.float32)
        output1_tensor = reference[output1_name].astype(np.float32)

        report_path = find_single(f"**/bonsai-lowbit-persistent-onnx-segment-{segment[0]}-{segment[1]}/report.json", SEGMENT_ROOT)
        segment_report = load_json(report_path)
        onnx_name = segment_report.get("onnx_path")
        require(isinstance(onnx_name, str), f"bad onnx_path in segment report: {segment_report}")
        onnx_path = find_single(f"**/bonsai-lowbit-persistent-onnx-segment-{segment[0]}-{segment[1]}/{onnx_name}", SEGMENT_ROOT)
        require(sha256_file(onnx_path) == segment_report.get("onnx_sha256"), f"onnx sha mismatch: {onnx_path}")

        handoffs.append({
            "segment_block_indices": segment,
            "onnx_artifact_report_sha256": sha256_file(report_path),
            "onnx_sha256": segment_report.get("onnx_sha256"),
            "onnx_size_bytes": segment_report.get("onnx_size_bytes"),
            "input": {
                "source": "initial_hidden" if previous_block is None else f"segment{previous_block[0]}_{previous_block[1]}_block1_output",
                "array_name": input_name,
                "shape": list(input_tensor.shape),
                "dtype": str(input_tensor.dtype),
                "sha256_float32_le": sha256_array(input_tensor),
            },
            "outputs": {
                "block0_output": {
                    "array_name": output0_name,
                    "shape": list(output0_tensor.shape),
                    "dtype": str(output0_tensor.dtype),
                    "sha256_float32_le": sha256_array(output0_tensor),
                },
                "block1_output": {
                    "array_name": output1_name,
                    "shape": list(output1_tensor.shape),
                    "dtype": str(output1_tensor.dtype),
                    "sha256_float32_le": sha256_array(output1_tensor),
                },
            },
            "next_segment_input_array": output1_name if segment != SEGMENTS[-1] else None,
        })
        previous_output_name = output1_name
        previous_block = segment

    final = reference["expected_final_block19_output"].astype(np.float32)
    final_from_last = reference[previous_output_name].astype(np.float32)
    require(np.array_equal(final, final_from_last), "final expected block19 output does not match last segment block1 output")

    report = {
        "ok": True,
        "artifact_kind": "chain_state_handoff_report",
        "source_validation_report": str(VALIDATION_REPORT),
        "source_validation_report_sha256": sha256_file(VALIDATION_REPORT),
        "reference_report_sha256": sha256_file(find_single("**/report.json", REFERENCE_ROOT)),
        "reference_npz_sha256": sha256_file(ref_path),
        "graph_kind": "ten_by_two_single_blocks_modulated_attention_to_out_residual_stack",
        "sequence_block_count": 20,
        "onnx_segment_count": 10,
        "segment_block_indices": SEGMENTS,
        "persistent_onnx_artifacts": True,
        "reusable_onnx_chain_artifact": True,
        "validated_without_lowbit_source_reload": True,
        "validated_from_persisted_onnx_files": True,
        "handoff_count": len(handoffs),
        "handoff_schema": {
            "input": "initial hidden for segment 0_1 or previous segment block1 output for later segments",
            "output": "block1 output is the next segment input",
            "hash": "sha256 over contiguous little-endian float32 bytes as stored by numpy on the runner",
        },
        "handoffs": handoffs,
        "final_output": {
            "array_name": "expected_final_block19_output",
            "shape": list(final.shape),
            "dtype": str(final.dtype),
            "sha256_float32_le": sha256_array(final),
        },
        "is_single_monolithic_onnx": False,
        "is_real_transformer_block": False,
        "is_full_bonsai_pipeline": False,
        "claim": "ten_by_two_split_persistent_onnx_segments_chain_state_handoffs_documented_without_lowbit_source_reload_not_single_monolithic_onnx_not_real_transformer_block_or_full_bonsai_pipeline",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": report["ok"],
        "artifact_kind": report["artifact_kind"],
        "handoff_count": report["handoff_count"],
        "final_output_sha256": report["final_output"]["sha256_float32_le"],
    }, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"ok": False, "error": str(exc)}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        sys.exit(1)
