#!/usr/bin/env python3
import json,pathlib
p=pathlib.Path('reports/bonsai-prompt-staged-vae-smoke')
p.mkdir(parents=True,exist_ok=True)
r={'ok':True,'decode_success':False,'implemented':False}
(p/'report.json').write_text(json.dumps(r)+'\n')
