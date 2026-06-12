#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import yaml

PROMPT = "a small bonsai tree in a ceramic pot"


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value in (None, "") else value.lower() in {"1", "true", "yes", "on"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_summary(value: Any) -> dict[str, Any]:
    import numpy as np
    if hasattr(value, "detach"):
        value = value.detach().cpu().float().numpy()
    arr = np.asarray(value)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(arr.min()) if arr.size else None,
        "max": float(arr.max()) if arr.size else None,
        "mean": float(arr.mean()) if arr.size else None,
        "sha256_float32_le": sha256_bytes(arr.astype("float32").tobytes()) if arr.size else None,
    }


def stage(name: str, allowed_claim: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    start = time.time()
    try:
        payload = fn()
        return {"name": name, "status": "passed", "seconds": round(time.time() - start, 3), "claim_promotable_to_manifest": True, "allowed_claim": allowed_claim, **payload}
    except BaseException as exc:
        return {"name": name, "status": "failed", "seconds": round(time.time() - start, 3), "claim_promotable_to_manifest": False, "allowed_claim": allowed_claim, "error_type": type(exc).__name__, "error": str(exc)[:3000]}


def skip_stage(name: str, allowed_claim: str, reason: str) -> dict[str, Any]:
    return {"name": name, "status": "skipped", "seconds": 0.0, "claim_promotable_to_manifest": False, "allowed_claim": allowed_claim, "skip_reason": reason}


def load_config(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tokenizer_probe(model_ref: str) -> dict[str, Any]:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_ref, subfolder="tokenizer")
    encoded = tokenizer(PROMPT, return_tensors="pt")
    ids = encoded["input_ids"]
    out = {"component": "tokenizer", "prompt": PROMPT, "class_name": tokenizer.__class__.__name__, "vocab_size": getattr(tokenizer, "vocab_size", None), "model_max_length": getattr(tokenizer, "model_max_length", None), "input_ids_shape": list(ids.shape), "input_ids_sha256_int64_le": sha256_bytes(ids.detach().cpu().numpy().astype("int64").tobytes())}
    if "attention_mask" in encoded:
        mask = encoded["attention_mask"]
        out["attention_mask_shape"] = list(mask.shape)
        out["attention_mask_sha256_int64_le"] = sha256_bytes(mask.detach().cpu().numpy().astype("int64").tobytes())
    return out


def text_encoder_probe(model_ref: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_ref, subfolder="tokenizer")
    model = AutoModel.from_pretrained(model_ref, subfolder="text_encoder", torch_dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        output = model(**tokenizer(PROMPT, return_tensors="pt"))
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None and isinstance(output, tuple) and output:
        hidden = output[0]
    if hidden is None:
        raise RuntimeError("text encoder did not return hidden state")
    return {"component": "text_encoder", "class_name": model.__class__.__name__, "hidden_state": tensor_summary(hidden)}


def scheduler_probe(model_ref: str) -> dict[str, Any]:
    import torch
    import diffusers
    from huggingface_hub import hf_hub_download
    cfg = load_config(hf_hub_download(repo_id=model_ref, filename="scheduler/scheduler_config.json", repo_type="model"))
    class_name = cfg.get("_class_name")
    if not class_name:
        raise RuntimeError("scheduler config is missing _class_name")
    scheduler = getattr(diffusers, class_name).from_pretrained(model_ref, subfolder="scheduler")
    set_kwargs: dict[str, Any] = {}
    if "mu" in inspect.signature(scheduler.set_timesteps).parameters:
        set_kwargs["mu"] = 0.0
    scheduler.set_timesteps(2, **set_kwargs)
    sample = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    step_kwargs: dict[str, Any] = {}
    if "return_dict" in inspect.signature(scheduler.step).parameters:
        step_kwargs["return_dict"] = True
    result = scheduler.step(torch.ones_like(sample) * 0.01, scheduler.timesteps[0], sample, **step_kwargs)
    prev = getattr(result, "prev_sample", None)
    if prev is None and isinstance(result, tuple) and result:
        prev = result[0]
    if prev is None:
        raise RuntimeError("scheduler.step did not return prev_sample")
    return {"component": "scheduler", "class_name": class_name, "set_timesteps_kwargs": set_kwargs, "timesteps": [float(x) for x in scheduler.timesteps.detach().cpu().float().tolist()[:2]], "prev_sample": tensor_summary(prev)}


def vae_probe(model_ref: str) -> dict[str, Any]:
    import torch
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(model_ref, subfolder="vae", torch_dtype=torch.float32)
    vae.eval()
    latents = torch.zeros((1, int(getattr(vae.config, "latent_channels", 4)), 8, 8), dtype=torch.float32)
    with torch.no_grad():
        decoded = vae.decode(latents, return_dict=False)[0]
    return {"component": "vae_decoder", "class_name": vae.__class__.__name__, "input_shape": list(latents.shape), "decoded_sample": tensor_summary(decoded)}


def transformer_config_probe(model_ref: str) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download
    cfg = load_config(hf_hub_download(repo_id=model_ref, filename="transformer/config.json", repo_type="model"))
    keys = ["in_channels", "out_channels", "num_layers", "num_single_layers", "attention_head_dim", "num_attention_heads", "joint_attention_dim", "pooled_projection_dim", "guidance_embeds", "axes_dims_rope"]
    return {"component": "transformer", "load_kind": "config_only", "class_name": cfg.get("_class_name"), "selected_config": {k: cfg.get(k) for k in keys if k in cfg}}


def transformer_load_probe(model_ref: str) -> dict[str, Any]:
    import torch
    import diffusers
    from huggingface_hub import hf_hub_download
    cfg = load_config(hf_hub_download(repo_id=model_ref, filename="transformer/config.json", repo_type="model"))
    cls = getattr(diffusers, cfg.get("_class_name", "FluxTransformer2DModel"))
    model = cls.from_pretrained(model_ref, subfolder="transformer", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    dtype_counts: dict[str, int] = {}
    param_count = 0
    for param in model.parameters():
        param_count += int(param.numel())
        dtype_counts[str(param.dtype)] = dtype_counts.get(str(param.dtype), 0) + int(param.numel())
    return {"component": "transformer", "load_kind": "weights", "class_name": cls.__name__, "param_count": param_count, "param_count_billion": round(param_count / 1_000_000_000, 3), "dtype_param_counts": dtype_counts}


def dependency_blocked_stage(name: str, allowed_claim: str, dependencies: dict[str, str]) -> dict[str, Any]:
    failed = {k: v for k, v in dependencies.items() if v != "passed"}
    if failed:
        return {"name": name, "status": "blocked", "seconds": 0.0, "claim_promotable_to_manifest": False, "allowed_claim": allowed_claim, "blocked_by": failed}
    return {"name": name, "status": "not_implemented", "seconds": 0.0, "claim_promotable_to_manifest": False, "allowed_claim": allowed_claim, "reason": "composed execution has not been implemented in this probe yet"}


def run(args: argparse.Namespace) -> int:
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_ref = cfg["candidate"]["model_ref"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_transformer_load = env_bool("BONSAI_RUN_TRANSFORMER_LOAD", False)
    require_all = env_bool("BONSAI_REQUIRE_ALL_COMBINED_CLAIMS", False)
    start = time.time()
    if run_transformer_load:
        stages = [
            stage("transformer_config_load", "bonsai_transformer_config_boundary_verified_not_runtime_execution", lambda: transformer_config_probe(model_ref)),
            stage("transformer_weight_load", "bonsai_real_transformer_weight_load_verified_not_onnx_execution", lambda: transformer_load_probe(model_ref)),
        ]
    else:
        stages = [
            stage("tokenizer_execution", "bonsai_tokenizer_execution_verified", lambda: tokenizer_probe(model_ref)),
            stage("text_encoder_execution", "bonsai_text_encoder_execution_verified", lambda: text_encoder_probe(model_ref)),
            stage("scheduler_execution", "bonsai_scheduler_step_execution_verified", lambda: scheduler_probe(model_ref)),
            stage("vae_execution", "bonsai_vae_decoder_execution_verified", lambda: vae_probe(model_ref)),
            stage("transformer_config_load", "bonsai_transformer_config_boundary_verified_not_runtime_execution", lambda: transformer_config_probe(model_ref)),
            skip_stage("transformer_weight_load", "bonsai_real_transformer_weight_load_verified_not_onnx_execution", "set BONSAI_RUN_TRANSFORMER_LOAD=true to attempt heavy transformer weight load"),
        ]
    statuses = {s["name"]: s["status"] for s in stages}
    stages.append(dependency_blocked_stage("full_pipeline_composition", "bonsai_full_pipeline_composition_verified", {"tokenizer_execution": statuses.get("tokenizer_execution", "not_run"), "text_encoder_execution": statuses.get("text_encoder_execution", "not_run"), "scheduler_execution": statuses.get("scheduler_execution", "not_run"), "vae_execution": statuses.get("vae_execution", "not_run"), "transformer_weight_load": statuses.get("transformer_weight_load")}))
    stages.append(dependency_blocked_stage("prompt_to_image_generation", "bonsai_prompt_to_image_generation_verified", {"full_pipeline_composition": stages[-1]["status"]}))
    stages.append(dependency_blocked_stage("single_monolithic_multi_block_onnx", "bonsai_single_monolithic_multi_block_onnx_verified", {"full_pipeline_composition": stages[-2]["status"]}))
    promotable = [s for s in stages if s.get("claim_promotable_to_manifest")]
    failed = [s for s in stages if s.get("status") == "failed"]
    report = {"experiment_id": "bonsai-combined-component-probe-v1", "model_ref": model_ref, "status": "passed" if promotable and not (require_all and failed) else "failed", "ci_conclusion": "success" if not (require_all and failed) else "failure", "claim_promotable_to_manifest": bool(promotable), "require_all": require_all, "run_transformer_load": run_transformer_load, "promotable_claims": [{"name": s["name"], "allowed_claim": s["allowed_claim"]} for s in promotable], "stages": stages, "seconds": round(time.time() - start, 3)}
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 1 if report["ci_conclusion"] == "failure" else 0


def self_test() -> int:
    assert env_bool("MISSING", True) is True
    assert sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    print("combined component probe self-test passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/bonsai-onnx-smoke.yml")
    parser.add_argument("--out-dir", default="reports/bonsai-combined-component-probe")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    return self_test() if args.self_test else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
