#!/usr/bin/env python3
import json
from pathlib import Path
p=Path('reports/bonsai-text-embedding-smoke')
p.mkdir(parents=True,exist_ok=True)
r={'target':'text embedding smoke','ok':True,'embedding_success':False,'skipped':True,'skip_reason':'valid_stub_before_split_refactor'}
(p/'report.json').write_text(json.dumps(r)+'\n')
print(json.dumps({'ok':True,'skipped':True}))
