#!/usr/bin/env python3
import json,pathlib,traceback
from bonsai_prompt_vae_core import run

p=pathlib.Path('reports/bonsai-prompt-staged-vae-smoke')
p.mkdir(parents=True,exist_ok=True)
r={'ok':True,'implemented':True}
try:
    r.update(run(p))
except Exception as e:
    r.update({'decode_success':False,'error':type(e).__name__+': '+str(e),'traceback':traceback.format_exc()[-2000:]})
(p/'report.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'ok':r.get('ok'),'decode_success':r.get('decode_success'),'error':r.get('error')},indent=2))
