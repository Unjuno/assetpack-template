#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def write_report(out_dir: Path, report: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_cos_model(path: Path, elem_type: int, patched: bool) -> dict:
    import onnx
    from onnx import TensorProto, helper

    path.parent.mkdir(parents=True, exist_ok=True)
    input_info = helper.make_tensor_value_info("x", elem_type, [1, 4])
    output_info = helper.make_tensor_value_info("y", elem_type, [1, 4])
    if patched:
        nodes = [
            helper.make_node("Cast", ["x"], ["x_float32"], name="cast_input_to_float32", to=TensorProto.FLOAT),
            helper.make_node("Cos", ["x_float32"], ["y_float32"], name="cos_float32"),
            helper.make_node("Cast", ["y_float32"], ["y"], name="cast_output_to_original", to=elem_type),
        ]
    else:
        nodes = [helper.make_node("Cos", ["x"], ["y"], name="cos_original")]
    graph = helper.make_graph(nodes, path.stem, [input_info], [output_info])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
    # Keep the IR version conservative for older onnxruntime wheels.
    model.ir_version = 8
    onnx.save(model, str(path))
    return {
        "path": str(path),
        "elem_type": TensorProto.DataType.Name(int(elem_type)),
        "patched": patched,
        "node_types": [node.op_type for node in nodes],
        "opset": 17,
        "ir_version": int(model.ir_version),
    }


def try_load(path: Path) -> dict:
    import onnxruntime as ort

    started = time.time()
    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(path), sess_options=sess_options, providers=["CPUExecutionProvider"])
        return {
            "status": "passed",
            "providers": session.get_providers(),
            "available_providers": ort.get_available_providers(),
            "inputs": [{"name": arg.name, "shape": list(arg.shape), "type": arg.type} for arg in session.get_inputs()],
            "outputs": [{"name": arg.name, "shape": list(arg.shape), "type": arg.type} for arg in session.get_outputs()],
            "execution_attempted": False,
            "seconds": round(time.time() - started, 6),
        }
    except BaseException as exc:
        return {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "execution_attempted": False,
            "seconds": round(time.time() - started, 6),
        }


def run(out_dir: str) -> int:
    started = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "experiment_id": "ort-cpu-kernel-probe-v2",
        "purpose": "Synthetic ONNX Runtime CPUExecutionProvider kernel probe for Cos FLOAT16 and FLOAT cast workaround.",
        "claim_promotable_to_manifest": False,
        "allowed_claim": "synthetic_ort_cpu_cos_float16_cast_workaround_evidence_not_bonsai_pipeline",
    }
    try:
        import onnx
        import onnxruntime as ort
        from onnx import TensorProto

        report["versions"] = {
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "available_providers": ort.get_available_providers(),
        }
        models = {
            "cos_float16_original": make_cos_model(out / "cos_float16_original.onnx", TensorProto.FLOAT16, patched=False),
            "cos_float32_original": make_cos_model(out / "cos_float32_original.onnx", TensorProto.FLOAT, patched=False),
            "cos_float16_with_float32_cast_patch": make_cos_model(out / "cos_float16_with_float32_cast_patch.onnx", TensorProto.FLOAT16, patched=True),
        }
        loads = {name: try_load(Path(meta["path"])) for name, meta in models.items()}
        report["models"] = models
        report["onnxruntime_loads"] = loads
        fp16_original_failed = loads["cos_float16_original"]["status"] == "failed"
        fp32_original_passed = loads["cos_float32_original"]["status"] == "passed"
        fp16_patched_passed = loads["cos_float16_with_float32_cast_patch"]["status"] == "passed"
        report["inference"] = {
            "cos_float16_original_load_failed": fp16_original_failed,
            "cos_float32_original_load_passed": fp32_original_passed,
            "cos_float16_with_float32_cast_patch_load_passed": fp16_patched_passed,
            "supports_bonsai_interpretation": fp16_original_failed and fp32_original_passed and fp16_patched_passed,
        }
        if fp16_original_failed and fp32_original_passed and fp16_patched_passed:
            report.update({
                "status": "passed",
                "ci_conclusion": "success",
                "claim_promotable_to_manifest": True,
                "allowed_claim": "synthetic_ort_cpu_cos_float16_cast_workaround_verified_not_bonsai_pipeline",
            })
        else:
            report.update({
                "status": "failed",
                "ci_conclusion": "success_with_probe_failure",
            })
    except BaseException as exc:
        report.update({
            "status": "failed",
            "ci_conclusion": "success_with_probe_failure",
            "error_type": type(exc).__name__,
            "error": str(exc)[:4000],
        })
    report["seconds"] = round(time.time() - started, 6)
    write_report(out, report)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="reports/ort-cpu-kernel-probe")
    args = parser.parse_args()
    return run(args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
