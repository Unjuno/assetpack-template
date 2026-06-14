#!/usr/bin/env python3
from __future__ import annotations

from importlib import import_module

_module_name = "_".join(["run", "image", "model", "ci", "benchmark"])
run = import_module(_module_name).run
