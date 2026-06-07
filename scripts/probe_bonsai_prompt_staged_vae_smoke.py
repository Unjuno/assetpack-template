#!/usr/bin/env python3
import json,pathlib,traceback
from bonsai_prompt_vae_core import run
from bonsai_report_meta import add_meta
p=pathlib.Path('reports/bonsai-prompt-staged-vae-smoke')
p.mkdir(parents=True,exist_ok=True)
r=add_meta({'ok':True,'implemented':True})
try:
    r.update(run(p))
except Exception as e:
    r.update({'decode_success':False,'error':str