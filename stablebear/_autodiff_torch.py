"""PyTorch adapter for stablebear._autodiff."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch


class _Op(torch.autograd.Function):
    @staticmethod
    def forward(ctx, primal, vjp, *params):
        values = [to_numpy(p) for p in params]
        out = primal(*values)
        ctx.vjp = vjp
        ctx.values = values
        ctx.param_meta = [(p.dtype, p.device) for p in params]
        return torch.from_numpy(np.ascontiguousarray(out)).to(params[0].device)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grads = ctx.vjp(to_numpy(grad_outputs[0]), *ctx.values)
        return (
            None,
            None,
            *(torch.as_tensor(g, dtype=dtype, device=device) for g, (dtype, device) in zip(grads, ctx.param_meta)),
        )


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def is_view(array: Any, x: torch.Tensor) -> bool:
    """Whether a NumPy array is the memory of the tensor, so that it has the
    values of the tensor without comparing them."""
    return (
        isinstance(array, np.ndarray) and x.device.type == "cpu" and x.is_contiguous() and array.flags.c_contiguous
        and array.size == x.numel() and array.dtype.itemsize == x.element_size()
        and array.__array_interface__["data"][0] == x.data_ptr())


def requires_grad(x: torch.Tensor) -> bool:
    return bool(x.requires_grad)


def version(x: torch.Tensor) -> int:
    """Changes when `x` is changed in place."""
    return x._version


def as_tensor(x: Any, like: torch.Tensor | None = None) -> torch.Tensor:
    if like is None:
        return torch.tensor(x)
    return torch.tensor(x, dtype=like.dtype, device=like.device)


def as_bool_tensor(x: Any, like: torch.Tensor) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.bool, device=like.device)


def gather(values: torch.Tensor, idx: np.ndarray) -> torch.Tensor:
    return values[torch.as_tensor(idx, device=values.device)]


def concat(xs: list) -> torch.Tensor:
    return torch.cat(xs)


def cumsum(x: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(x, dim=0)


def segment_sums(x: torch.Tensor, segments: np.ndarray, count: int) -> torch.Tensor:
    """The sums of the entries of x by segment: entry k is the sum of the
    entries i with segments[i] == k, for ``count`` segments."""
    index = torch.as_tensor(segments, dtype=torch.long, device=x.device)
    return torch.zeros(count, dtype=x.dtype, device=x.device).index_add(0, index, x)


def segment_max(x: torch.Tensor, segments: np.ndarray, count: int) -> torch.Tensor:
    """The largest of the entries of x by segment, and 0 for a segment
    without entries or with negative entries only: entry k is the largest
    entry i with segments[i] == k, for ``count`` segments."""
    index = torch.as_tensor(segments, dtype=torch.long, device=x.device)
    return torch.zeros(count, dtype=x.dtype, device=x.device).scatter_reduce(0, index, x, "amax")


def max_along(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.amax(x, dim=dim)


def row_norms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((x * x).sum(dim=1))


def stack_columns(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.stack([a, b], dim=1)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b


def transpose(a: torch.Tensor) -> torch.Tensor:
    return a.T


def solve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.linalg.solve(a, b)


def cholesky(a: torch.Tensor) -> torch.Tensor:
    return torch.linalg.cholesky(a)


def row_lp_norms(x: torch.Tensor, p: float) -> torch.Tensor:
    if p == float("inf"):
        return x.abs().max(dim=1).values
    return (x.abs() ** p).sum(dim=1) ** (1 / p)


def reshape(x: torch.Tensor, shape: tuple) -> torch.Tensor:
    return x.reshape(shape)


def apply(primal: Callable[..., Any], vjp: Callable[..., Any], *params: Any) -> torch.Tensor:
    """The result of ``primal(*values)`` for the NumPy values of the
    tensors, as a tensor whose backward pass takes the gradients of the
    values from ``vjp(grad_out, *values)``."""
    params = tuple(p if isinstance(p, torch.Tensor) else torch.as_tensor(p) for p in params)
    return _Op.apply(primal, vjp, *params)
