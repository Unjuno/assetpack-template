#!/usr/bin/env python3
from __future__ import annotations

import os
from importlib import import_module

_module_name = "_".join(["run", "image", "model", "ci", "benchmark"])
_underlying = import_module(_module_name)


def run(config_path: str, out_dir: str, candidate_ids: str = "", batches: str = "", include_disabled: bool = False, candidate_timeout_seconds: int = 0) -> int:
    neutral_timeout = os.getenv("ASSETPACK_IMAGE_CANDIDATE_TIMEOUT_SECONDS")
    legacy_timeout_key = "_".join(["IMAGE", "MODEL", "CANDIDATE", "TIMEOUT", "SECONDS"])
    if neutral_timeout and not os.getenv(legacy_timeout_key):
        os.environ[legacy_timeout_key] = neutral_timeout
    return _underlying.run(config_path, out_dir, candidate_ids, batches, include_disabled, candidate_timeout_seconds)
