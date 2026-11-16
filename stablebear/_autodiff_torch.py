"""PyTorch adapter for stablebear._autodiff."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch


class _Op(torch.autograd.Function):
    @staticmethod
    def forward(ctx, primal, vjp, *params):
        values = [to_numpy(p) for p in params]
        out, ctx.residuals = primal(*values)
        ctx.vjp = vjp
        ctx.values = values
        ctx.param_meta = [(p.dtype, p.device) for p in params]
        return torch.from_numpy(np.ascontiguousarray(out)).to(params[0].device)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grads = ctx.vjp(to_numpy(grad_outputs[0]), ctx.residuals, *ctx.values)
        return (
            None,
            None,
            *(torch.as_tensor(g, dtype=dtype, device=device) for g, (dtype, device) in zip(grads, ctx.param_meta)),
        )


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def concrete(x: torch.Tensor) -> np.ndarray:
    return to_numpy(x)


def as_tensor(x: Any, like: torch.Tensor | None = None) -> torch.Tensor:
    if like is None:
        return torch.tensor(x)
    return torch.tensor(x, dtype=like.dtype, device=like.device)


def gather(values: torch.Tensor, idx: np.ndarray) -> torch.Tensor:
    return values[torch.as_tensor(idx, device=values.device)]


def concat(xs: list) -> torch.Tensor:
    return torch.cat(xs)


def sort(x: torch.Tensor) -> torch.Tensor:
    return torch.sort(x).values


def apply(
    primal: Callable[..., Any], vjp: Callable[..., Any], *params: Any, out: Any = None, residuals: Any = ()
) -> torch.Tensor:
    """``primal(*values)`` returns (result, residuals), and
    ``vjp(grad_out, residuals, *values)`` the gradients of the values."""
    params = tuple(p if isinstance(p, torch.Tensor) else torch.as_tensor(p) for p in params)
    return _Op.apply(primal, vjp, *params)
