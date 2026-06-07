#!/usr/bin/env python3
import json,pathlib
from bonsai_prompt_vae_core import run
from bonsai_report_meta import add_meta
p=pathlib.Path('reports/bonsai-prompt-staged-vae-smoke')
p.mkdir(parents=True,exist_ok=True)
r=add_meta({'ok':True,'implemented':True})
r.update(run(p))
(p/'report.json').write_text(json.dumps(r,indent=2)+'\n')
