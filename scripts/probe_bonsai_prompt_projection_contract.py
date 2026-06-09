#!/usr/bin/env python3
import json,pathlib,torch
from bonsai_prompt_projection import project_prompt_hidden
from bonsai_text_consts import TEXT_HIDDEN_DIM
p=pathlib.Path('reports/bonsai-prompt-projection-contract')
p.mkdir(parents=True,exist_ok=True)
ok=False; err=None
try:
    project_prompt_hidden(torch.zeros(1,2,TEXT_HIDDEN_DIM))
except NotImplementedError as e:
    ok=True; err=str(e)
r={'ok':ok,'projection_wired':False,'expected_error':'NotImplementedError','error':err}
(p/'report.json').write_text(json.dumps(r)+'\n')
