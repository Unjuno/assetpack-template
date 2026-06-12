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
VALIDATION_REPORT = ROOT / "bonsai-lowbit-ten-by-two-split-persistent-onnx-artifacts-validation" / "report.json"
HANDOFF_REPORT = ROOT / "bonsai-lowbit-ten-by-two-chain-handoff" / "report.json"
OUT = ROOT / "bonsai-lowbit-ten-by-two-input-boundary" / "report.json"
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


def tensor_info_from_numpy(name: str, array: np.ndarray) -> dict[str, Any]:
    arr = np.ascontiguousarray(array.astype(np.float32, copy=False))
    return {
        "name": name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256_float32_le": hashlib.sha256(arr.tobytes()).hexdigest(),
    }


def io_info(value: Any) -> dict[str, Any]:
    return {
        "name": value.name,
        "shape": list(value.shape),
        "type": value.type,
    }


def main() -> None:
    validation = load_json(VALIDATION_REPORT)
    require(validation.get("ok") is True, f"validation report is not ok: {validation}")
    require(validation.get("persistent_onnx_artifacts") is True, f"validation did not use persistent artifacts: {validation}")
    require(validation.get("validated_without_lowbit_source_reload") is True, f"validation reloaded low-bit source: {validation}")
    require(validation.get("validated_from_persisted_onnx_files") is True, f"validation did not use persisted ONNX files: {validation}")

    handoff = load_json(HANDOFF_REPORT)
    require(handoff.get("ok") is True, f"handoff report is not ok: {handoff}")
    require(handoff.get("handoff_count") == 10, f"unexpected handoff count: {handoff}")

    ref_report = load_json(find_single("**/report.json", REFERENCE_ROOT))
    ref_path = find_single("**/reference.npz", REFERENCE_ROOT)
    require(sha256_file(ref_path) == ref_report.get("reference_sha256"), f"reference sha256 mismatch: {ref_path}")
    reference = np.load(ref_path)
    initial_hidden = reference["initial_hidden"].astype(np.float32)
    temb = reference["temb"].astype(np.float32)

    hidden_dim = None
    semantic_to_out_width = None
    sequence_length = int(initial_hidden.shape[1])
    boundary_segments: list[dict[str, Any]] = []

    for segment in SEGMENTS:
        report_path = find_single(f"**/bonsai-lowbit-persistent-onnx-segment-{segment[0]}-{segment[1]}/report.json", SEGMENT_ROOT)
        report = load_json(report_path)
        require(report.get("artifact_kind") == "persistent_onnx_segment", f"bad segment report: {report}")
        require(report.get("block_indices") == segment, f"bad segment block indices: {report}")
        require(report.get("critical_outputs_allclose_rtol_1e_4_atol_1e_5") is True, f"segment not allclose: {report}")
        onnx_name = report.get("onnx_path")
        require(isinstance(onnx_name, str) and onnx_name.endswith(".onnx"), f"bad onnx path: {report}")
        onnx_path = find_single(f"**/bonsai-lowbit-persistent-onnx-segment-{segment[0]}-{segment[1]}/{onnx_name}", SEGMENT_ROOT)
        require(sha256_file(onnx_path) == report.get("onnx_sha256"), f"segment sha mismatch: {onnx_path}")

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inputs = [io_info(item) for item in session.get_inputs()]
        outputs = [io_info(item) for item in session.get_outputs()]
        input_names = [item["name"] for item in inputs]
        output_names = [item["name"] for item in outputs]
        require(input_names == ["hidden", "temb"], f"unexpected inputs for segment {segment}: {input_names}")
        require(output_names == [
            "block0_output",
            "block1_output",
            "block0_semantic_to_out_input",
            "block1_semantic_to_out_input",
            "block0_gate",
            "block1_gate",
            "block0_weights",
            "block1_weights",
        ], f"unexpected outputs for segment {segment}: {output_names}")

        if hidden_dim is None:
            hidden_dim = int(report.get("hidden_dim"))
        if semantic_to_out_width is None:
            semantic_to_out_width = int(report.get("semantic_input_width"))
        require(int(report.get("hidden_dim")) == hidden_dim, f"hidden_dim mismatch in {report_path}")
        require(int(report.get("semantic_input_width")) == semantic_to_out_width, f"semantic width mismatch in {report_path}")
        require(report.get("is_real_transformer_block") is False, f"must not claim real transformer block: {report}")
        require(report.get("is_full_bonsai_pipeline") is False, f"must not claim full pipeline: {report}")

        boundary_segments.append({
            "segment_block_indices": segment,
            "onnx_report_sha256": sha256_file(report_path),
            "onnx_sha256": report.get("onnx_sha256"),
            "onnx_size_bytes": report.get("onnx_size_bytes"),
            "onnx_inputs": inputs,
            "onnx_outputs": outputs,
            "external_inputs": ["hidden", "temb"],
            "diagnostic_internal_outputs": [
                "block0_semantic_to_out_input",
                "block1_semantic_to_out_input",
                "block0_gate",
                "block1_gate",
                "block0_weights",
                "block1_weights",
            ],
        })

    report = {
        "ok": True,
        "artifact_kind": "input_boundary_report",
        "source_validation_report": str(VALIDATION_REPORT),
        "source_validation_report_sha256": sha256_file(VALIDATION_REPORT),
        "source_handoff_report": str(HANDOFF_REPORT),
        "source_handoff_report_sha256": sha256_file(HANDOFF_REPORT),
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
        "input_boundary": {
            "external_onnx_inputs": [
                tensor_info_from_numpy("hidden", initial_hidden),
                tensor_info_from_numpy("temb", temb),
            ],
            "hidden_source": "synthetic reference tensor used to probe the low-bit single-stream block chain",
            "temb_source": "synthetic modulation tensor used to exercise shared single-stream modulation",
            "prompt_tokens_present": False,
            "text_encoder_present": False,
            "scheduler_present": False,
            "vae_present": False,
            "image_latents_present": False,
            "real_bonsai_pipeline_inputs_present": False,
        },
        "internal_boundary": {
            "hidden_dim": hidden_dim,
            "sequence_length": sequence_length,
            "semantic_to_out_internal_width": semantic_to_out_width,
            "semantic_to_out_internal_width_note": "The historical field name semantic_input_width describes the internal to_out input width emitted by the qkv/mlp projection diagnostic path; it is not evidence of an external text encoder or prompt-token input.",
            "modulation_input_width": hidden_dim,
            "modulation_output_width": 3 * int(hidden_dim),
        },
        "segments": boundary_segments,
        "is_single_monolithic_onnx": False,
        "is_real_transformer_block": False,
        "is_full_bonsai_pipeline": False,
        "forbidden_claims_not_verified": [
            "full Bonsai ONNX pipeline",
            "real transformer block ONNX verification",
            "prompt-to-image generation verification",
            "single monolithic multi-block ONNX when the verified result is segmented",
        ],
        "claim": "ten_by_two_split_persistent_onnx_segments_input_boundary_documented_hidden_temb_only_no_prompt_text_encoder_scheduler_vae_or_full_pipeline",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": report["ok"],
        "artifact_kind": report["artifact_kind"],
        "external_onnx_inputs": [item["name"] for item in report["input_boundary"]["external_onnx_inputs"]],
        "semantic_to_out_internal_width": report["internal_boundary"]["semantic_to_out_internal_width"],
        "real_bonsai_pipeline_inputs_present": report["input_boundary"]["real_bonsai_pipeline_inputs_present"],
    }, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"ok": False, "error": str(exc)}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        sys.exit(1)
