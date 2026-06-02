#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict, quantized_prefixes

OUT_DIR = Path("reports/bonsai-lowbit-quantized-inventory")


def classify_prefix(prefix: str) -> str:
    if re.match(r"^single_transformer_blocks\.\d+\.attn\.to_qkv_mlp_proj$", prefix):
        return "single_block_fused_qkv_mlp"
    if re.match(r"^single_transformer_blocks\.\d+\.attn\.to_out$", prefix):
        return "single_block_to_out"
    if re.match(r"^transformer_blocks\.\d+\.attn\.to_q$", prefix):
        return "double_block_img_to_q"
    if re.match(r"^transformer_blocks\.\d+\.attn\.to_k$", prefix):
        return "double_block_img_to_k"
    if re.match(r"^transformer_blocks\.\d+\.attn\.to_v$", prefix):
        return "double_block_img_to_v"
    if re.match(r"^transformer_blocks\.\d+\.attn\.to_out\.0$", prefix):
        return "double_block_img_to_out"
    if re.match(r"^transformer_blocks\.\d+\.attn\.add_q_proj$", prefix):
        return "double_block_txt_to_q"
    if re.match(r"^transformer_blocks\.\d+\.attn\.add_k_proj$", prefix):
        return "double_block_txt_to_k"
    if re.match(r"^transformer_blocks\.\d+\.attn\.add_v_proj$", prefix):
        return "double_block_txt_to_v"
    if re.match(r"^transformer_blocks\.\d+\.attn\.to_add_out$", prefix):
        return "double_block_txt_to_out"
    if re.match(r"^transformer_blocks\.\d+\.ff\.linear_in$", prefix):
        return "double_block_img_ff_in"
    if re.match(r"^transformer_blocks\.\d+\.ff\.linear_out$", prefix):
        return "double_block_img_ff_out"
    if re.match(r"^transformer_blocks\.\d+\.ff_context\.linear_in$", prefix):
        return "double_block_txt_ff_in"
    if re.match(r"^transformer_blocks\.\d+\.ff_context\.linear_out$", prefix):
        return "double_block_txt_ff_out"
    return "other_quantized"


def family(prefix: str) -> str:
    if prefix.startswith("single_transformer_blocks."):
        return "single_transformer_blocks"
    if prefix.startswith("transformer_blocks."):
        return "transformer_blocks"
    return "other"


def block_index(prefix: str) -> int | None:
    m = re.match(r"^(?:single_transformer_blocks|transformer_blocks)\.(\d+)\.", prefix)
    return None if m is None else int(m.group(1))


def tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def prefix_entry(sd: dict[str, Any], prefix: str) -> dict[str, Any]:
    orig_shape = [int(v) for v in sd[f"{prefix}.orig_shape"].tolist()]
    metadata = [int(v) for v in sd[f"{prefix}.metadata"].tolist()]
    packed_nbytes = sum(tensor_nbytes(sd[f"{prefix}.{name}"]) for name in ["W_q", "scales", "zeros", "orig_shape", "metadata"])
    expanded_fp32_nbytes = int(orig_shape[0] * orig_shape[1] * 4)
    return {
        "prefix": prefix,
        "family": family(prefix),
        "block_index": block_index(prefix),
        "class": classify_prefix(prefix),
        "orig_shape": orig_shape,
        "metadata": {"nbits": int(metadata[1]), "group_size": int(metadata[2]), "raw": metadata},
        "packed_nbytes": packed_nbytes,
        "expanded_fp32_nbytes": expanded_fp32_nbytes,
        "expanded_to_packed_ratio": expanded_fp32_nbytes / packed_nbytes if packed_nbytes else None,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lowbit_path, sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    prefixes = quantized_prefixes(sd)
    entries = [prefix_entry(sd, p) for p in prefixes]
    class_counts = Counter(e["class"] for e in entries)
    family_counts = Counter(e["family"] for e in entries)
    by_family_block: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for e in entries:
        by_family_block[e["family"]][str(e["block_index"] if e["block_index"] is not None else "none")].append(e["class"])

    unexpected = [e for e in entries if e["class"] == "other_quantized"]
    single_count = sum(1 for e in entries if e["family"] == "single_transformer_blocks")
    double_count = sum(1 for e in entries if e["family"] == "transformer_blocks")
    total_packed_nbytes = sum(e["packed_nbytes"] for e in entries)
    total_expanded_fp32_nbytes = sum(e["expanded_fp32_nbytes"] for e in entries)
    report = {
        "source_model_ref": LOWBIT_REF,
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "lowbit_path": str(lowbit_path),
        "quantized_prefix_count": len(entries),
        "single_transformer_quantized_count": single_count,
        "transformer_blocks_quantized_count": double_count,
        "other_quantized_count": len(unexpected),
        "class_counts": dict(sorted(class_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "block_class_counts": {fam: {idx: Counter(classes) for idx, classes in blocks.items()} for fam, blocks in by_family_block.items()},
        "total_packed_nbytes": total_packed_nbytes,
        "total_expanded_fp32_nbytes": total_expanded_fp32_nbytes,
        "total_expanded_to_packed_ratio": total_expanded_fp32_nbytes / total_packed_nbytes if total_packed_nbytes else None,
        "unexpected_quantized_prefixes": unexpected,
        "entries": entries,
        "all_quantized_prefixes_classified": not unexpected,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=lambda o: dict(o)) + "\n", encoding="utf-8")
    print(json.dumps({
        "quantized_prefix_count": report["quantized_prefix_count"],
        "single_transformer_quantized_count": report["single_transformer_quantized_count"],
        "transformer_blocks_quantized_count": report["transformer_blocks_quantized_count"],
        "other_quantized_count": report["other_quantized_count"],
        "all_quantized_prefixes_classified": report["all_quantized_prefixes_classified"],
        "class_counts": report["class_counts"],
        "total_expanded_to_packed_ratio": report["total_expanded_to_packed_ratio"],
    }, indent=2))
    return 0 if report["all_quantized_prefixes_classified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
