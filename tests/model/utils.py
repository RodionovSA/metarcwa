import torch
from dataclasses import fields, is_dataclass

def iter_tensors(obj, prefix=""):
    if is_dataclass(obj):
        for f in fields(obj):
            yield from iter_tensors(getattr(obj, f.name), f"{prefix}{f.name}.")
    elif isinstance(obj, torch.Tensor):
        yield prefix.rstrip("."), obj
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            yield from iter_tensors(item, f"{prefix}{i}.")

def describe_tensors(spec) -> None:
    for name, t in iter_tensors(spec):
        print(f"{name:30s} shape={t.shape} device={t.device}  dtype={t.dtype}  "
              f"requires_grad={t.requires_grad}  grad_fn={t.grad_fn}")