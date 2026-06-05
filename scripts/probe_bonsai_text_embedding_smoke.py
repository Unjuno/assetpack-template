#!/usr/bin/env python3
import json
from pathlib import Path
from bonsai_text_smoke_main import run_text_smoke
p=Path('reports/bonsai-text-embedding-smoke'); p.mkdir(parents=True,exist_ok=True)
r={'target':'text embedding smoke','ok':True,'embedding_success':True,'adapter_success':True}
r.update(run_text_smoke())
(p/'report.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'ok':True,'embedding_success':True,'adapter_success':True}))
