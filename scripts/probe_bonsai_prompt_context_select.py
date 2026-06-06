#!/usr/bin/env python3
import json
from pathlib import Path
from bonsai_prompt_context import load_prompt_context
from bonsai_prompt_context_select import select_staged_context
from bonsai_tensor_stats import tensor_stat
p=Path('reports/bonsai-prompt-context-select'); p.mkdir(parents=True,exist_ok=True)
ctx=load_prompt_context(); sel=select_staged_context(ctx)
r={'target':'prompt context select','ok':True,'source':tensor_stat(ctx),'selected':tensor_stat(sel)}
(p