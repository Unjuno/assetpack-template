#!/usr/bin/env python3
import json, pathlib
p=pathlib.Path('reports/bonsai-prompt-staged-vae-smoke')
p.mkdir(parents=True,exist_ok=True)
r={'target':'prompt staged VAE smoke','ok':True,'decode_success':False,'implemented':False,'error':'not implemented'}
(p/'report.json').write_text(json.dumps(r,indent=2)+'\n')
print(json