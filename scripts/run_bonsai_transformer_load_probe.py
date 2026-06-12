#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
import time
from pathlib import Path
from typing import Callable, TypeVar

import yaml

T = TypeVar("T")


FLOAT_ELEM_TYPES = {1, 10, 11, 16}  # FLOAT, FLOAT16, DOUBLE, BFLOAT16


def load_config(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    return default if value in (None, "") else value.lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return int(value)


def write_report(out_dir: Path, report: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def retry_operation(label: str, fn: Callable[[], T], attempts: int, sleep_seconds: int, report: dict) -> T:
    records = []
    report.setdefault("retry_records", {})[label] = records
    for attempt in range(1, attempts + 1):
        try:
            result = fn()
            records.append({"attempt": attempt, "status": "passed"})
            return result
        except BaseException as exc:
            records.append({
                "attempt": attempt,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            })
            if attempt >= attempts:
                raise
            time.sleep(sleep_seconds)
    raise RuntimeError(f"unreachable retry state for {label}")


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


def node_record(node, value_types: dict[str, int]) -> dict:
    return {
        "name": node.name,
        "op_type": node.op_type,
        "input_types": [tensor_elem_type_name(value_types.get(name)) for name in node.input],
        "output_types": [tensor_elem_type_name(value_types.get(name)) for name in node.output],
        "inputs": list(node.input),
        "outputs": list(node.output),
    }


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
    non_float_focus_nodes = []
    target_node_names = {"node_cos_1", "node_sin_1"}
    target_nodes = []
    producer_by_output = {}
    consumers_by_input = {}
    for node in typed_model.graph.node:
        for output in node.output:
            producer_by_output[output] = node
        for input_name in node.input:
            consumers_by_input.setdefault(input_name, []).append(node)
    for node in typed_model.graph.node:
        if node.op_type not in focus_ops:
            continue
        record = node_record(node, value_types)
        focus_nodes.append(record)
        input_elem_types = [value_types.get(name) for name in node.input]
        if any(elem_type is not None and elem_type not in FLOAT_ELEM_TYPES for elem_type in input_elem_types):
            non_float_focus_nodes.append(record)
        if node.name in target_node_names:
            producers = [producer_by_output.get(name) for name in node.input]
            consumers = []
            for output in node.output:
                consumers.extend(consumers_by_input.get(output, []))
            target_nodes.append({
                "node": record,
                "input_producers": [node_record(prod, value_types) for prod in producers if prod is not None],
                "output_consumers": [node_record(consumer, value_types) for consumer in consumers[:20]],
            })
    diagnostics["focus_ops"] = list(focus_ops)
    diagnostics["focus_nodes"] = focus_nodes[:100]
    diagnostics["focus_node_count"] = len(focus_nodes)
    diagnostics["non_float_focus_nodes"] = non_float_focus_nodes[:100]
    diagnostics["non_float_focus_node_count"] = len(non_float_focus_nodes)
    diagnostics["target_nodes"] = target_nodes
    diagnostics["focus_node_type_counts"] = {}
    for node in focus_nodes:
        for elem_type in node["input_types"]:
            key = elem_type if elem_type is not None else "UNKNOWN"
            diagnostics["focus_node_type_counts"][key] = diagnostics["focus_node_type_counts"].get(key, 0) + 1
    return diagnostics


def patch_trig_input_cast_for_ort_cpu(src_path: Path, patched_path: Path, cast_back: bool) -> dict:
    import onnx
    from onnx import TensorProto, helper, shape_inference

    model = onnx.load(str(src_path), load_external_data=False)
    try:
        typed_model = shape_inference.infer_shapes(model, strict_mode=False, data_prop=False)
        shape_inference_status = "passed"
    except BaseException as exc:
        typed_model = model
        shape_inference_status = "failed"
        shape_inference_error = {"error_type": type(exc).__name__, "error": str(exc)[:1000]}
    value_types = collect_value_types(typed_model)
    new_nodes = []
    patched_nodes = []
    for index, node in enumerate(model.graph.node):
        input_elem_type = value_types.get(node.input[0]) if len(node.input) == 1 else None
        output_elem_type = value_types.get(node.output[0]) if len(node.output) == 1 else None
        should_patch = (
            node.op_type in {"Cos", "Sin"}
            and len(node.input) == 1
            and len(node.output) == 1
            and input_elem_type is not None
            and input_elem_type not in FLOAT_ELEM_TYPES
        )
        if not should_patch:
            new_nodes.append(node)
            continue
        base_name = node.name or f"{node.op_type.lower()}_{index}"
        original_input = node.input[0]
        original_output = node.output[0]
        float_input = f"{original_input}__{base_name}_float32"
        float_output = f"{original_output}__{base_name}_float32"
        cast_in = helper.make_node(
            "Cast",
            inputs=[original_input],
            outputs=[float_input],
            name=f"{base_name}_cast_input_to_float32",
            to=TensorProto.FLOAT,
        )
        trig_output = float_output if cast_back else original_output
        trig = helper.make_node(
            node.op_type,
            inputs=[float_input],
            outputs=[trig_output],
            name=f"{base_name}_float32",
        )
        nodes_to_add = [cast_in, trig]
        patch_kind = "non_float_trig_input_cast_to_float32"
        if cast_back:
            cast_out = helper.make_node(
                "Cast",
                inputs=[float_output],
                outputs=[original_output],
                name=f"{base_name}_cast_output_back_to_original_type",
                to=output_elem_type or input_elem_type,
            )
            nodes_to_add.append(cast_out)
            patch_kind = "non_float_trig_input_cast_to_float32_and_output_cast_back"
        new_nodes.extend(nodes_to_add)
        patched_nodes.append({
            "name": node.name,
            "op_type": node.op_type,
            "input": original_input,
            "output": original_output,
            "input_type": tensor_elem_type_name(input_elem_type),
            "output_type": tensor_elem_type_name(output_elem_type),
            "patch": patch_kind,
        })
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    onnx.save_model(model, str(patched_path))
    report = {
        "status": "passed",
        "source_path": str(src_path),
        "patched_path": str(patched_path),
        "patch_kind": "non_float_trig_cast_for_onnxruntime_cpu_load",
        "cast_back": cast_back,
        "shape_inference_status": shape_inference_status,
        "patched_node_count": len(patched_nodes),
        "patched_nodes": patched_nodes[:100],
        "external_data_reused_from_source": True,
    }
    if shape_inference_status == "failed":
        report.update(shape_inference_error)
    return report


def make_ort_session(onnx_path: Path):
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )


def ort_session_metadata(session) -> dict:
    import onnxruntime as ort

    return {
        "providers": session.get_providers(),
        "available_providers": ort.get_available_providers(),
        "inputs": [node_arg_metadata(arg) for arg in session.get_inputs()],
        "outputs": [node_arg_metadata(arg) for arg in session.get_outputs()],
        "graph_optimization_level": "ORT_DISABLE_ALL",
        "execution_attempted": False,
    }


def run(config_path: str, out_dir: str) -> int:
    start = time.time()
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    model_ref = cfg["candidate"]["model_ref"]
    out = Path(out_dir)
    export_attempt = env_bool("BONSAI_RUN_TRANSFORMER_ONNX_EXPORT", False)
    ort_load_attempt = env_bool("BONSAI_RUN_TRANSFORMER_ONNXRUNTIME_LOAD", False)
    trig_cast_patch_attempt = env_bool("BONSAI_RUN_TRANSFORMER_ONNXRUNTIME_TRIG_CAST_PATCH", False)
    hf_retry_attempts = env_int("BONSAI_HF_RETRY_ATTEMPTS", 3)
    hf_retry_sleep_seconds = env_int("BONSAI_HF_RETRY_SLEEP_SECONDS", 20)
    report = {
        "experiment_id": "bonsai-transformer-load-probe-v8",
        "model_ref": model_ref,
        "download_weights": True,
        "runtime_load": True,
        "hf_retry_attempts": hf_retry_attempts,
        "hf_retry_sleep_seconds": hf_retry_sleep_seconds,
        "onnx_export_attempted": export_attempt,
        "onnx_export_revision": os.getenv("BONSAI_TRANSFORMER_ONNX_EXPORT_REVISION", ""),
        "onnxruntime_load_attempted": ort_load_attempt,
        "onnxruntime_load_revision": os.getenv("BONSAI_TRANSFORMER_ONNXRUNTIME_LOAD_REVISION", ""),
        "onnxruntime_trig_cast_patch_attempted": trig_cast_patch_attempt,
        "onnxruntime_trig_cast_patch_revision": os.getenv("BONSAI_TRANSFORMER_ONNXRUNTIME_TRIG_CAST_PATCH_REVISION", ""),
        "claim_promotable_to_manifest": False,
        "allowed_claim": "bonsai_real_transformer_weight_load_verified_not_onnx_execution",
    }
    try:
        import torch
        import diffusers
        from huggingface_hub import hf_hub_download

        config_path = retry_operation(
            "hf_hub_download_transformer_config",
            lambda: hf_hub_download(repo_id=model_ref, filename="transformer/config.json", repo_type="model"),
            hf_retry_attempts,
            hf_retry_sleep_seconds,
            report,
        )
        transformer_config = load_config(config_path)
        class_name = transformer_config.get("_class_name", "FluxTransformer2DModel")
        cls = getattr(diffusers, class_name)
        model = retry_operation(
            "diffusers_transformer_from_pretrained",
            lambda: cls.from_pretrained(model_ref, subfolder="transformer", torch_dtype=torch.float16, low_cpu_mem_usage=True),
            hf_retry_attempts,
            hf_retry_sleep_seconds,
            report,
        )
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
                original_ort_load_passed = False
                try:
                    session = make_ort_session(onnx_path)
                    original_ort_load_passed = True
                    report["onnxruntime_load"] = {
                        "status": "passed",
                        "claim_promotable_to_manifest": True,
                        "allowed_claim": "bonsai_transformer_minimal_onnxruntime_load_verified_not_execution",
                        "path": str(onnx_path),
                        **ort_session_metadata(session),
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
                        "diagnostic_hint": "Inspect onnx_graph_diagnostics.non_float_focus_nodes and target_nodes for Cos/Sin input dtypes, especially node_cos_1.",
                        "seconds": round(time.time() - ort_load_start, 3),
                    }
                    report["ci_conclusion"] = "success_with_probe_failure"
                if trig_cast_patch_attempt:
                    patch_attempts = {}
                    if original_ort_load_passed:
                        report["onnxruntime_trig_cast_patch"] = {
                            "status": "skipped",
                            "reason": "Original ONNX Runtime CPU load passed; patch not needed.",
                            "claim_promotable_to_manifest": False,
                            "execution_attempted": False,
                        }
                    else:
                        for label, cast_back in {
                            "input_cast_float_output": False,
                            "input_cast_and_output_cast_back": True,
                        }.items():
                            patch_start = time.time()
                            try:
                                patched_path = out / f"transformer_minimal_ort_cpu_trig_{label}.onnx"
                                patch_report = patch_trig_input_cast_for_ort_cpu(onnx_path, patched_path, cast_back=cast_back)
                                patched_session = make_ort_session(patched_path)
                                patched_files = sorted(p.name for p in out.glob(f"transformer_minimal_ort_cpu_trig_{label}.onnx*"))
                                patch_attempts[label] = {
                                    "status": "passed",
                                    "claim_promotable_to_manifest": True,
                                    "allowed_claim": "bonsai_transformer_minimal_onnxruntime_cpu_load_with_non_float_trig_cast_patch_verified_not_execution",
                                    "source_path": str(onnx_path),
                                    "patched_path": str(patched_path),
                                    "files": patched_files,
                                    "patch": patch_report,
                                    "onnxruntime_load": ort_session_metadata(patched_session),
                                    "seconds": round(time.time() - patch_start, 3),
                                }
                            except BaseException as patch_exc:
                                patch_attempts[label] = {
                                    "status": "failed",
                                    "claim_promotable_to_manifest": False,
                                    "allowed_claim": "bonsai_transformer_minimal_onnxruntime_cpu_load_with_non_float_trig_cast_patch_verified_not_execution",
                                    "execution_attempted": False,
                                    "error_type": type(patch_exc).__name__,
                                    "error": str(patch_exc)[:4000],
                                    "seconds": round(time.time() - patch_start, 3),
                                }
                        any_patch_passed = any(item.get("status") == "passed" for item in patch_attempts.values())
                        report["onnxruntime_trig_cast_patch"] = {
                            "status": "passed" if any_patch_passed else "failed",
                            "claim_promotable_to_manifest": any_patch_passed,
                            "allowed_claim": "bonsai_transformer_minimal_onnxruntime_cpu_load_with_non_float_trig_cast_patch_verified_not_execution",
                            "attempts": patch_attempts,
                            "execution_attempted": False,
                        }
                        if not any_patch_passed:
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
