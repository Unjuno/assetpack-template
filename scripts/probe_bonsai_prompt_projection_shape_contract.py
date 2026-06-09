#!/usr/bin/env python3
import json,pathlib,torch
from bonsai_prompt_projection_deterministic import deterministic_project_prompt_hidden
from bonsai_text_consts import BONSAI_CONTEXT_DIM,TEXT_HIDDEN_DIM

def bad(x):
    try:
        deterministic_project_prompt_hidden(x); return False
    except ValueError:
        return True
p=pathlib.Path('reports/bonsai-prompt-projection-shape-contract')
p.mkdir(parents=True,exist_ok=True)
x=torch.zeros(2,3,TEXT_HIDDEN_DIM)
y=deterministic_project_prompt_hidden(x)
br=bad(torch.zeros(3,TEXT_HIDDEN_DIM))
bd=bad(torch.zeros(2,3,TEXT_HIDDEN_DIM+1))
r={'ok':list(y.shape)==[2,3,BONSAI_CONTEXT_DIM] and bool(torch.isfinite(y).all()) and br and bd,'shape':list(y.shape),'finite':bool(torch.isfinite(y).all()),'bad_rank':br,'bad_dim':bd,'projection':'deterministic_repeat_stub'}
(p/'report.json').write_text(json.dumps(r)+'\n')
