#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
import time
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    return default if value in (None, "") else value.lower() in {"1", "true", "yes", "on"}


def write_report(out_dir: Path, report: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def node_arg_metadata(arg) -> dict:
    return {
        "name": arg.name,
        "shape": list(arg.shape) if arg.shape is not None else None,
        "type": arg.type,
    }


def tensor_elem_type_name(elem_type: int | None) -> str | None:
    if elem_type is None:
        return None
    try:
        import onnx

        return onnx.TensorProto.DataType.Name(int(elem_type))
    except BaseException:
        return str(elem_type)


def value_info_elem_type(value_info) -> int | None:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("elem_type"):
        return None
    return int(tensor_type.elem_type)


def collect_value_types(model) -> dict[str, int]:
    value_types: dict[str, int] = {}
    for value_info in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        elem_type = value_info_elem_type(value_info)
        if elem_type is not None:
            value_types[value_info.name] = elem_type
    for initializer in model.graph.initializer:
        value_types[initializer.name] = int(initializer.data_type)
    return value_types


def onnx_graph_diagnostics(onnx_path: Path, focus_ops: tuple[str, ...] = ("Cos", "Sin")) -> dict:
    import onnx
    from onnx import shape_inference

    diagnostics: dict = {
        "path": str(onnx_path),
        "load_external_data": False,
        "shape_inference_attempted": True,
    }
    model = onnx.load(str(onnx_path), load_external_data=False)
    diagnostics["opset_imports"] = [
        {"domain": opset.domain, "version": int(opset.version)} for opset in model.opset_import
    ]
    diagnostics["ir_version"] = int(model.ir_version)
    diagnostics["node_count"] = len(model.graph.node)
    diagnostics["initializer_count"] = len(model.graph.initializer)
    diagnostics["external_initializer_count"] = sum(
        1 for initializer in model.graph.initializer if initializer.data_location == onnx.TensorProto.EXTERNAL
    )
    diagnostics["op_type_counts"] = {}
    for node in model.graph.node:
        diagnostics["op_type_counts"][node.op_type] = diagnostics["op_type_counts"].get(node.op_type, 0) + 1
    try:
        inferred = shape_inference.infer_shapes(model, strict_mode=False, data_prop=False)
        diagnostics["shape_inference_status"] = "passed"
        typed_model = inferred
    except BaseException as exc:
        diagnostics["shape_inference_status"] = "failed"
        diagnostics["shape_inference_error_type"] = type(exc).__name__
        diagnostics["shape_inference_error"] = str(exc)[:1000]
        typed_model = model
    value_types = collect_value_types(typed_model)
    focus_nodes = []
    for node in typed_model.graph.node:
        if node.op_type not in focus_ops:
            continue
        focus_nodes.append({
            "name": node.name,
            "op_type": node.op_type,
            "input_types": [tensor_elem_type_name(value_types.get(name)) for name in node.input],
            "output_types": [tensor_elem_type_name(value_types.get(name)) for name in node.output],
            "inputs": list(node.input),
            "outputs": list(node.output),
        })
    diagnostics["focus_ops"] = list(focus_ops)
    diagnostics["focus_nodes"] = focus_nodes[:100]
    diagnostics["focus_node_count"] = len(focus_nodes)
    diagnostics["focus_node_type_counts"] = {}
    for node in focus_nodes:
        for elem_type in node["input_types"]:
            key = elem_type if elem_type is not None else "UNKNOWN"
            diagnostics["focus_node_type_counts"][key] = diagnostics["focus_node_type_counts"].get(key, 0) + 1
    return diagnostics


def run(config_path: str, out_dir: str) -> int:
    start = time.time()
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    model_ref = cfg["candidate"]["model_ref"]
    out = Path(out_dir)
    export_attempt = env_bool("BONSAI_RUN_TRANSFORMER_ONNX_EXPORT", False)
    ort_load_attempt = env_bool("BONSAI_RUN_TRANSFORMER_ONNXRUNTIME_LOAD", False)
    report = {
        "experiment_id": "bonsai-transformer-load-probe-v5",
        "model_ref": model_ref,
        "download_weights": True,
        "runtime_load": True,
        "onnx_export_attempted": export_attempt,
        "onnx_export_revision": os.getenv("BONSAI_TRANSFORMER_ONNX_EXPORT_REVISION", ""),
        "onnxruntime_load_attempted": ort_load_attempt,
        "onnxruntime_load_revision": os.getenv("BONSAI_TRANSFORMER_ONNXRUNTIME_LOAD_REVISION", ""),
        "claim_promotable_to_manifest": False,
        "allowed_claim": "bonsai_real_transformer_weight_load_verified_not_onnx_execution",
    }
    try:
        import torch
        import diffusers
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(repo_id=model_ref, filename="transformer/config.json", repo_type="model")
        transformer_config = load_config(config_path)
        class_name = transformer_config.get("_class_name", "FluxTransformer2DModel")
        cls = getattr(diffusers, class_name)
        model = cls.from_pretrained(model_ref, subfolder="transformer", torch_dtype=torch.float16, low_cpu_mem_usage=True)
        model.eval()
        param_count = 0
        dtype_counts = {}
        for param in model.parameters():
            count = int(param.numel())
            param_count += count
            dtype_counts[str(param.dtype)] = dtype_counts.get(str(param.dtype), 0) + count
        report.update({
            "status": "passed",
            "ci_conclusion": "success",
            "claim_promotable_to_manifest": True,
            "class_name": class_name,
            "param_count": param_count,
            "param_count_billion": round(param_count / 1000000000, 3),
            "dtype_param_counts": dtype_counts,
        })
        write_report(out, {**report, "seconds": round(time.time() - start, 3), "partial_before_onnx_export": True})
        if export_attempt:
            class Wrapper(torch.nn.Module):
                def __init__(self, inner):
                    super().__init__()
                    self.inner = inner

                def forward(self, hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids):
                    result = self.inner(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, timestep=timestep, img_ids=img_ids, txt_ids=txt_ids, return_dict=False)
                    return result[0]

            axes = transformer_config.get("axes_dims_rope", [32, 32, 32, 32])
            axes_len = len(axes) if isinstance(axes, list) else 4
            in_channels = int(transformer_config.get("in_channels", 128))
            joint_dim = int(transformer_config.get("joint_attention_dim", 7680))
            example_inputs = (
                torch.zeros((1, 1, in_channels), dtype=torch.float16),
                torch.zeros((1, 1, joint_dim), dtype=torch.float16),
                torch.zeros((1,), dtype=torch.float16),
                torch.zeros((1, axes_len), dtype=torch.float32),
                torch.zeros((1, axes_len), dtype=torch.float32),
            )
            onnx_path = out / "transformer_minimal.onnx"
            kwargs = {
                "input_names": ["hidden_states", "encoder_hidden_states", "timestep", "img_ids", "txt_ids"],
                "output_names": ["sample"],
                "opset_version": 17,
                "do_constant_folding": False,
            }
            export_params = inspect.signature(torch.onnx.export).parameters
            if "external_data" in export_params:
                kwargs["external_data"] = True
            elif "use_external_data_format" in export_params:
                kwargs["use_external_data_format"] = True
            torch.onnx.export(Wrapper(model), example_inputs, str(onnx_path), **kwargs)
            extra_files = sorted(p.name for p in out.glob("transformer_minimal.onnx*"))
            report.update({
                "onnx_export": {
                    "status": "passed",
                    "claim_promotable_to_manifest": True,
                    "allowed_claim": "bonsai_transformer_minimal_onnx_export_verified_not_full_pipeline",
                    "path": str(onnx_path),
                    "size_bytes": onnx_path.stat().st_size,
                    "files": extra_files,
                    "external_data_enabled": "external_data" in kwargs or "use_external_data_format" in kwargs,
                }
            })
            try:
                report["onnx_graph_diagnostics"] = onnx_graph_diagnostics(onnx_path)
            except BaseException as diag_exc:
                report["onnx_graph_diagnostics"] = {
                    "status": "failed",
                    "error_type": type(diag_exc).__name__,
                    "error": str(diag_exc)[:1000],
                }
            write_report(out, {**report, "seconds": round(time.time() - start, 3), "partial_before_onnxruntime_load": True})
            if ort_load_attempt:
                ort_load_start = time.time()
                try:
                    import onnxruntime as ort

                    sess_options = ort.SessionOptions()
                    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                    sess_options.intra_op_num_threads = 1
                    sess_options.inter_op_num_threads = 1
                    session = ort.InferenceSession(
                        str(onnx_path),
                        sess_options=sess_options,
                        providers=["CPUExecutionProvider"],
                    )
                    report["onnxruntime_load"] = {
                        "status": "passed",
                        "claim_promotable_to_manifest": True,
                        "allowed_claim": "bonsai_transformer_minimal_onnxruntime_load_verified_not_execution",
                        "path": str(onnx_path),
                        "providers": session.get_providers(),
                        "available_providers": ort.get_available_providers(),
                        "inputs": [node_arg_metadata(arg) for arg in session.get_inputs()],
                        "outputs": [node_arg_metadata(arg) for arg in session.get_outputs()],
                        "graph_optimization_level": "ORT_DISABLE_ALL",
                        "execution_attempted": False,
                        "seconds": round(time.time() - ort_load_start, 3),
                    }
                except BaseException as ort_exc:
                    report["onnxruntime_load"] = {
                        "status": "failed",
                        "claim_promotable_to_manifest": False,
                        "allowed_claim": "bonsai_transformer_minimal_onnxruntime_load_verified_not_execution",
                        "path": str(onnx_path),
                        "execution_attempted": False,
                        "error_type": type(ort_exc).__name__,
                        "error": str(ort_exc)[:4000],
                        "likely_failure_boundary": "onnxruntime_session_initialization_kernel_resolution",
                        "diagnostic_hint": "Inspect onnx_graph_diagnostics.focus_nodes for Cos/Sin input dtypes; CPUExecutionProvider may lack kernels for specific low-precision element types.",
                        "seconds": round(time.time() - ort_load_start, 3),
                    }
                    report["ci_conclusion"] = "success_with_probe_failure"
        elif ort_load_attempt:
            report["onnxruntime_load"] = {
                "status": "blocked",
                "claim_promotable_to_manifest": False,
                "allowed_claim": "bonsai_transformer_minimal_onnxruntime_load_verified_not_execution",
                "reason": "BONSAI_RUN_TRANSFORMER_ONNX_EXPORT must be true before ONNX Runtime load can be probed.",
                "execution_attempted": False,
            }
            report["ci_conclusion"] = "success_with_probe_failure"
    except BaseException as exc:
        report.update({
            "ci_conclusion": "success_with_probe_failure",
            "error_type": type(exc).__name__,
            "error": str(exc)[:4000],
        })
    report["seconds"] = round(time.time() - start, 3)
    write_report(out, report)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/bonsai-onnx-smoke.yml")
    parser.add_argument("--out-dir", default="reports/bonsai-transformer-load-probe")
    args = parser.parse_args()
    return run(args.config, args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
