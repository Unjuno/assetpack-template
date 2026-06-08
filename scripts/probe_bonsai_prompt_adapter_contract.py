#!/usr/bin/env python3
import json,pathlib,torch
from bonsai_prompt_adapter import adapt_prompt_hidden
from bonsai_text_consts import BONSAI_CONTEXT_DIM,TEXT_HIDDEN_DIM
p=pathlib.Path('reports/bonsai-prompt-adapter-contract')
p.mkdir(parents=True,exist_ok=True)
x=torch.zeros(2,3,TEXT_HIDDEN_DIM)
y=adapt_prompt_hidden(x)
r={'ok':list(y.shape)==[2,3,BONSAI_CONTEXT_DIM],'shape':list(y.shape),'adapter':'repeat_2560_to_7680'}
(p/'report.json').write_text(json.dumps(r)+'\n')
