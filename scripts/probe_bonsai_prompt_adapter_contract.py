#!/usr/bin/env python3
import json,pathlib,torch
from bonsai_prompt_adapter import adapt_prompt_hidden
from bonsai_prompt_adapter_guard import require_repeat_adapter
from bonsai_prompt_adapter_kind import adapter_name
from bonsai_text_consts import BONSAI_CONTEXT_DIM,TEXT_HIDDEN_DIM

def bad(x):
    try:
        adapt_prompt_hidden(x); return False
    except ValueError:
        return True
p=pathlib.Path('reports/bonsai-prompt-adapter-contract')
p.mkdir(parents=True,exist_ok=True)
k=require_repeat_adapter()
x=torch.zeros(2,3,TEXT_HIDDEN_DIM)
y=adapt_prompt_hidden(x)
br=bad(torch.zeros(3,TEXT_HIDDEN_DIM))
bd=bad(torch.zeros(2,3,TEXT_HIDDEN_DIM+1))
r={'ok':list(y.shape)==[2,3,BONSAI_CONTEXT_DIM] and br and bd and k=='repeat','shape':list(y.shape),'bad_rank':br,'bad_dim':bd,'adapter_kind':k,'adapter':adapter_name()}
(p/'report.json').write_text(json.dumps(r)+'\n')
