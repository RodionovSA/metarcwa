# metarcwa/model/utils.py
# DESCRIPTION

import torch
import torch.nn as nn

def register(module, name, value, dtype=torch.float32):
    """Register `value` on `module` under `name`.

    If `value` is an nn.Parameter it becomes an optimizable parameter;
    otherwise it is stored as a (non-gradient) buffer that still moves
    with .to() and is saved in state_dict().
    """
    if isinstance(value, nn.Parameter):
        setattr(module, name, value)
    else:
        module.register_buffer(name, torch.as_tensor(value, dtype=dtype))