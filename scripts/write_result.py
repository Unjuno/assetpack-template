#!/usr/bin/env python3
import argparse, json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--out-dir',required=True); p.add_argument('--outcome',default=''); a=p.parse_args()
out=Path(a.out_dir)
req=json.loads((out/'request.json').read_text()) if (out/'request.json').exists() else {}
rep=json.loads((out/'report.json').read_text()) if (out/'report.json').exists() else {}
miss=req.get('missing_terms') or []
ok=rep.get('summary',{}).get('passed',0)>0
if miss:
    body='## Asset request not generated\n\nMissing required terms:\n'+''.join(f'- `{x}`\n' for x in miss)
elif ok:
    body='## Asset image generated\n\n- image: available in the workflow artifact\n'
else:
    body='## Asset image generation incomplete\n\n- reason: `'+str(rep.get('reason') or 'see report.json')+'`\n'
out.mkdir(parents=True,exist_ok=True)
(out/'generation-comment.md').write_text(body,encoding='utf-8')
