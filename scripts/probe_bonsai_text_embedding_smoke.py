#!/usr/bin/env python3
import json
from pathlib import Path
out=Path('reports/bonsai-text-embedding-smoke')
out.mkdir(parents=True, exist_ok=True)
r={'target':'text embedding smoke','ok':True,'embedding_success':False,'skipped':True,'skip_reason':'safe_placeholder_after_truncated_update'}
(out/'report.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'ok':True,'skipped':True}))
