#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
import platform
import resource
import shutil
import time
from pathlib import Path
from typing import Any

import yaml


def now_seconds() -> float:
    return time.time()


def max_rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 3)


def disk_snapshot(path: Path) -> dict[str, int]:
    usage = shutil.disk_usage(path)
    return {
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_array(value: Any) -> str:
    import numpy as np

    arr = np.asarray(value)
    return sha256_bytes(arr.tobytes())


def write_report(out_dir: Path, report: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def comma_set(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def select_candidates(cfg: dict, candidate_ids: set[str], batches: set[str]) -> list[dict]:
    selected = []
    for candidate in cfg.get("candidates", []):
        if candidate_ids and candidate.get("id") not in candidate_ids:
            continue
        if batches and candidate.get("batch") not in batches:
            continue
        selected.append(candidate)
    return selected


def public_candidate(candidate: dict) -> dict:
    keys = [
        "id",
        "enabled",
        "batch",
        "ci_stage",
        "method",
        "model_ref",
        "base_model_ref",
        "lora_model_ref",
        "pipeline_class",
        "scheduler_class",
        "height",
        "width",
        "steps",
        "guidance_scale",
        "expected_role",
        "license_hint",
        "notes",
    ]
    return {key: candidate.get(key) for key in keys if key in candidate}


def load_diffusers_class(name: str):
    import diffusers

    return getattr(diffusers, name)


def torch_dtype_from_cfg(cfg: dict):
    import torch

    dtype_name = cfg.get("runtime", {}).get("torch_dtype", "float32")
    return getattr(torch, dtype_name)


def load_pipeline(candidate: dict, cfg: dict):
    method = candidate.get("method")
    if method in {"openvino_text_to_image", "stable_diffusion_cpp_text_to_image", "component_load_placeholder"}:
        return None, {
            "status": "skipped",
            "reason": f"Method {method} is not a diffusers pipeline for generic ONNX probing.",
        }
    pipeline_class = candidate.get("pipeline_class", "DiffusionPipeline")
    cls = load_diffusers_class(pipeline_class)
    if method in {"diffusers_lora_text_to_image", "diffusers_lora_load_only"}:
        scheduler_class = candidate.get("scheduler_class", "LCMScheduler")
        scheduler_cls = load_diffusers_class(scheduler_class)
        pipe = cls.from_pretrained(candidate["base_model_ref"], torch_dtype=torch_dtype_from_cfg(cfg))
        pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
        pipe.load_lora_weights(candidate["lora_model_ref"])
    else:
        pipe = cls.from_pretrained(candidate["model_ref"], torch_dtype=torch_dtype_from_cfg(cfg))
    if hasattr(pipe, "safety_checker"):
        try:
            pipe.safety_checker = None
        except BaseException:
            pass
    if hasattr(pipe, "to"):
        pipe = pipe.to("cpu")
    return pipe, {"status": "passed", "pipeline_class": pipeline_class}


def component_summary(pipe: Any) -> dict:
    if pipe is None:
        return {}
    components = getattr(pipe, "components", {}) if hasattr(pipe, "components") else {}
    return {
        "pipeline_class": type(pipe).__name__,
        "component_keys": sorted(components.keys()),
        "has_unet": hasattr(pipe, "unet"),
        "has_transformer": hasattr(pipe, "transformer"),
        "has_vae": hasattr(pipe, "vae"),
        "unet_class": type(getattr(pipe, "unet", None)).__name__ if hasattr(pipe, "unet") else None,
        "transformer_class": type(getattr(pipe, "transformer", None)).__name__ if hasattr(pipe, "transformer") else None,
        "vae_class": type(getattr(pipe, "vae", None)).__name__ if hasattr(pipe, "vae") else None,
    }


def unet_export_supported(pipe: Any) -> tuple[bool, str]:
    if pipe is None or not hasattr(pipe, "unet"):
        return False, "Pipeline has no UNet component for generic UNet ONNX export."
    unet = pipe.unet
    config = getattr(unet, "config", None)
    if config is None:
        return False, "UNet has no config."
    addition_embed_type = getattr(config, "addition_embed_type", None)
    if addition_embed_type not in {None, "", "none"}:
        return False, f"UNet addition_embed_type={addition_embed_type!r} requires extra conditioning inputs not covered by the generic probe."
    cross_attention_dim = getattr(config, "cross_attention_dim", None)
    in_channels = getattr(config, "in_channels", None)
    if cross_attention_dim is None or in_channels is None:
        return False, "UNet missing cross_attention_dim or in_channels for dummy input construction."
    return True, "Generic Stable-Diffusion-style UNet export probe is supported."


def make_unet_dummy_inputs(pipe: Any, candidate: dict, cfg: dict):
    import torch

    unet = pipe.unet
    unet_config = unet.config
    height = int(candidate.get("height", 256))
    width = int(candidate.get("width", 256))
    latent_h = max(8, height // 8)
    latent_w = max(8, width // 8)
    in_channels = int(getattr(unet_config, "in_channels"))
    cross_attention_dim = int(getattr(unet_config, "cross_attention_dim"))
    seq_len = int(candidate.get("onnx_dummy_seq_len", 77))
    sample = torch.randn(1, in_channels, latent_h, latent_w, dtype=torch.float32)
    timestep = torch.tensor([1], dtype=torch.int64)
    encoder_hidden_states = torch.randn(1, seq_len, cross_attention_dim, dtype=torch.float32)
    return sample, timestep, encoder_hidden_states


def export_unet_to_onnx(pipe: Any, candidate: dict, cfg: dict, out_dir: Path) -> dict:
    import torch
    import onnx

    class UNetWrapper(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet

        def forward(self, sample, timestep, encoder_hidden_states):
            output = self.unet(sample, timestep, encoder_hidden_states, return_dict=False)
            return output[0]

    started = now_seconds()
    export_dir = out_dir / "onnx"
    export_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = export_dir / "unet_minimal.onnx"
    sample, timestep, encoder_hidden_states = make_unet_dummy_inputs(pipe, candidate, cfg)
    wrapper = UNetWrapper(pipe.unet).eval()
    input_names = ["sample", "timestep", "encoder_hidden_states"]
    output_names = ["latent"]
    kwargs = {
        "input_names": input_names,
        "output_names": output_names,
        "opset_version": int(candidate.get("onnx_opset", 17)),
        "do_constant_folding": False,
    }
    signature = inspect.signature(torch.onnx.export)
    if "external_data" in signature.parameters:
        kwargs["external_data"] = True
    elif "use_external_data_format" in signature.parameters:
        kwargs["use_external_data_format"] = True
    torch.onnx.export(wrapper, (sample, timestep, encoder_hidden_states), str(onnx_path), **kwargs)
    model = onnx.load(str(onnx_path), load_external_data=False)
    return {
        "status": "passed",
        "kind": "unet_minimal_onnx_export",
        "path": str(onnx_path),
        "files": sorted(path.name for path in export_dir.iterdir()),
        "size_bytes": int(onnx_path.stat().st_size),
        "opset_imports": [{"domain": item.domain, "version": int(item.version)} for item in model.opset_import],
        "ir_version": int(model.ir_version),
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "seconds": round(now_seconds() - started, 3),
        "dummy_inputs": {
            "sample_shape": list(sample.shape),
            "timestep_shape": list(timestep.shape),
            "encoder_hidden_states_shape": list(encoder_hidden_states.shape),
        },
    }


def load_and_run_onnx(onnx_path: Path, pipe: Any, candidate: dict, cfg: dict) -> dict:
    import numpy as np
    import onnxruntime as ort

    started = now_seconds()
    sample, timestep, encoder_hidden_states = make_unet_dummy_inputs(pipe, candidate, cfg)
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=["CPUExecutionProvider"])
    load_seconds = round(now_seconds() - started, 3)
    run_started = now_seconds()
    outputs = session.run(None, {
        "sample": sample.detach().cpu().numpy().astype(np.float32),
        "timestep": timestep.detach().cpu().numpy().astype(np.int64),
        "encoder_hidden_states": encoder_hidden_states.detach().cpu().numpy().astype(np.float32),
    })
    return {
        "status": "passed",
        "providers": session.get_providers(),
        "available_providers": ort.get_available_providers(),
        "inputs": [{"name": item.name, "shape": list(item.shape), "type": item.type} for item in session.get_inputs()],
        "outputs": [{"name": item.name, "shape": list(item.shape), "type": item.type} for item in session.get_outputs()],
        "load_seconds": load_seconds,
        "run_seconds": round(now_seconds() - run_started, 3),
        "total_seconds": round(now_seconds() - started, 3),
        "execution_attempted": True,
        "output_shapes": [list(output.shape) for output in outputs],
        "output_sha256": [sha256_array(output) for output in outputs],
    }


def probe_candidate(candidate: dict, cfg: dict, out_dir: Path) -> dict:
    started = now_seconds()
    record = {
        "candidate": public_candidate(candidate),
        "status": "started",
        "disk_before": disk_snapshot(Path.cwd()),
        "max_rss_mb_before": max_rss_mb(),
    }
    pipe = None
    try:
        import torch
        import diffusers

        torch.set_num_threads(int(cfg.get("runtime", {}).get("num_threads", 1)))
        load_started = now_seconds()
        pipe, load_record = load_pipeline(candidate, cfg)
        record["pipeline_load"] = {**load_record, "seconds": round(now_seconds() - load_started, 3)}
        record["components"] = component_summary(pipe)
        if pipe is None:
            record.update({
                "status": "skipped",
                "onnx_export": {"status": "skipped", "reason": load_record.get("reason")},
                "onnxruntime": {"status": "skipped", "reason": "No ONNX artifact was produced."},
                "seconds": round(now_seconds() - started, 3),
                "max_rss_mb_after_candidate": max_rss_mb(),
            })
            return record

        supported, reason = unet_export_supported(pipe)
        record["onnx_strategy"] = {
            "component": "unet",
            "supported_by_generic_probe": supported,
            "reason": reason,
        }
        if not supported:
            record.update({
                "status": "skipped",
                "onnx_export": {"status": "skipped", "reason": reason},
                "onnxruntime": {"status": "skipped", "reason": "No ONNX artifact was produced."},
                "seconds": round(now_seconds() - started, 3),
                "max_rss_mb_after_candidate": max_rss_mb(),
            })
            return record

        try:
            export_record = export_unet_to_onnx(pipe, candidate, cfg, out_dir)
            record["onnx_export"] = export_record
        except BaseException as exc:
            record.update({
                "status": "failed",
                "onnx_export": {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:4000],
                },
                "onnxruntime": {"status": "skipped", "reason": "ONNX export failed."},
                "seconds": round(now_seconds() - started, 3),
                "max_rss_mb_after_candidate": max_rss_mb(),
            })
            return record

        try:
            ort_record = load_and_run_onnx(Path(record["onnx_export"]["path"]), pipe, candidate, cfg)
            record["onnxruntime"] = ort_record
            record["status"] = "passed"
        except BaseException as exc:
            record["onnxruntime"] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc)[:4000],
                "execution_attempted": False,
            }
            record["status"] = "failed"

        record["versions"] = {
            "torch": torch.__version__,
            "diffusers": getattr(diffusers, "__version__", None),
        }
        record["seconds"] = round(now_seconds() - started, 3)
        record["max_rss_mb_after_candidate"] = max_rss_mb()
        return record
    except BaseException as exc:
        record.update({
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:4000],
            "seconds": round(now_seconds() - started, 3),
            "max_rss_mb_after_candidate": max_rss_mb(),
        })
        return record
    finally:
        try:
            del pipe
        except BaseException:
            pass
        gc.collect()
        record["disk_after"] = disk_snapshot(Path.cwd())


def run(config_path: str, out_dir: str, candidate_ids: str = "", batches: str = "") -> int:
    started = now_seconds()
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    out = Path(out_dir)
    candidate_id_set = comma_set(candidate_ids or os.getenv("IMAGE_MODEL_CANDIDATE_IDS", ""))
    batch_set = comma_set(batches or os.getenv("IMAGE_MODEL_CANDIDATE_BATCHES", ""))
    selected = select_candidates(cfg, candidate_id_set, batch_set)
    report = {
        "experiment_id": "image-model-onnx-feasibility-v1",
        "source_experiment_id": cfg.get("experiment_id"),
        "prompt": cfg.get("prompt"),
        "selection": {
            "candidate_ids": sorted(candidate_id_set),
            "batches": sorted(batch_set),
            "selected_count": len(selected),
        },
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cwd": str(Path.cwd()),
            "disk_start": disk_snapshot(Path.cwd()),
        },
        "claim_promotable_to_manifest": False,
        "allowed_claim": "ci_image_model_onnx_feasibility_measurement_not_full_pipeline",
        "status": "started",
        "candidates": [],
    }
    write_report(out, report)
    for candidate in selected:
        result = probe_candidate(candidate, cfg, out / "candidates" / candidate["id"])
        report["candidates"].append(result)
        passed = sum(1 for item in report["candidates"] if item.get("status") == "passed")
        failed = sum(1 for item in report["candidates"] if item.get("status") == "failed")
        skipped = sum(1 for item in report["candidates"] if item.get("status") == "skipped")
        report["summary_so_far"] = {"passed": passed, "failed": failed, "skipped": skipped, "total": len(report["candidates"])}
        report["seconds_so_far"] = round(now_seconds() - started, 3)
        write_report(out, report)
    passed = sum(1 for item in report["candidates"] if item.get("status") == "passed")
    failed = sum(1 for item in report["candidates"] if item.get("status") == "failed")
    skipped = sum(1 for item in report["candidates"] if item.get("status") == "skipped")
    report["summary"] = {"passed": passed, "failed": failed, "skipped": skipped, "total": len(report["candidates"])}
    report["status"] = "passed" if passed > 0 else "failed"
    report["ci_conclusion"] = "success" if failed == 0 and passed > 0 else "success_with_candidate_failures"
    report["seconds"] = round(now_seconds() - started, 3)
    report["system"]["disk_end"] = disk_snapshot(Path.cwd())
    report["system"]["max_rss_mb_end"] = max_rss_mb()
    write_report(out, report)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/image-model-ci-benchmark.yml")
    parser.add_argument("--out-dir", default="reports/image-model-onnx-feasibility")
    parser.add_argument("--candidate-ids", default="")
    parser.add_argument("--batches", default="")
    args = parser.parse_args()
    return run(args.config, args.out_dir, args.candidate_ids, args.batches)


if __name__ == "__main__":
    raise SystemExit(main())
