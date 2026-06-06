#!/usr/bin/env python3
import json
from pathlib import Path
from bonsai_prompt_context import load_prompt_context
from bonsai_prompt_context_select import select_staged_context
from bonsai_tensor_stats import tensor_stat
p=Path('reports/bonsai-prompt-context-select')
p.mkdir(parents=True,exist_ok=True)
sel=select_staged_context(load_prompt_context())
r={'target':'prompt context select','ok':True,'selected':tensor_stat(sel)}
(p/'report.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'ok':True,'shape':r['selected']['shape']}))
