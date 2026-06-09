import os
from pathlib import Path

import torch

from bonsai_text_consts import BONSAI_CONTEXT_DIM, TEXT_HIDDEN_DIM


def _load_state(path):
    obj = torch.load(path, map_location='cpu')
    state = obj.get('state_dict', obj) if isinstance(obj, dict) else obj
    if not isinstance(state, dict):
        raise ValueError(f'loaded projection is not a state dict: {type(state).__name__}')
    return state


def _require_shape(name, tensor, expected):
    if not hasattr(tensor, 'shape'):
        raise ValueError(f'projection {name} is not a tensor')
    actual = list(tensor.shape)
    if actual != expected:
        raise ValueError(f'projection {name} shape mismatch: expected={expected} actual={actual}')


def project_prompt_hidden(hidden):
    path = os.environ.get('BONSAI_PROMPT_PROJECTION_PATH')
    if not path:
        raise NotImplementedError('projection adapter is not wired')

    if hidden.ndim != 3:
        raise ValueError('hidden rank')
    if hidden.shape[-1] != TEXT_HIDDEN_DIM:
        raise ValueError('hidden dim')

    state = _load_state(Path(path))
    weight = state.get('weight')
    bias = state.get('bias')
    _require_shape('weight', weight, [BONSAI_CONTEXT_DIM, TEXT_HIDDEN_DIM])
    _require_shape('bias', bias, [BONSAI_CONTEXT_DIM])

    x = hidden.float()
    y = torch.matmul(x, weight.float().t()) + bias.float()
    return y
