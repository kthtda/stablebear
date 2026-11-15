"""Gradients of stablebear operations in other autodiff frameworks.

An operation opts in with the ``differentiable`` decorator, naming the
parameters that can carry gradients and a vector-Jacobian product:

    def _f_vjp(grad_out, x, p, verbose=False):
        return (d_x, d_p)   # gradient of sum(grad_out * f(x, p)) per name in wrt

    @differentiable("x", "p", vjp=_f_vjp)
    def f(x, p, verbose=False): ...

The function itself stays framework-free. A parameter is tracked when it is a
tensor of a supported framework, or a stablebear object that carries one, such
as a Pcf built by arithmetic with a tracked scalar. The decorator converts the
tracked parameters to NumPy (rebuilding stablebear objects from their values),
calls the function, and runs the call as an operation of that framework, so the
result carries gradients there. The vjp receives the same arguments as the
function and returns NumPy values; entries for untracked parameters are ignored.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, TypeVar

import numpy as np

F = TypeVar("F", bound=Callable[..., Any])


def _unwrap(x: Any) -> Any:
    """A Keras 3 variable as the variable or tensor of its backend: a
    tf.Variable, a torch parameter, or a JAX array. Anything else unchanged."""
    if type(x).__module__.split(".")[0] == "keras" and hasattr(x, "value"):
        return x.value
    return x


def _framework(x: Any) -> Any:
    """Return the adapter module for the framework that owns `x`, or None."""
    root = type(_unwrap(x)).__module__.split(".")[0]
    if root == "torch":
        from . import _autodiff_torch

        return _autodiff_torch
    if root in ("jax", "jaxlib"):
        from . import _autodiff_jax

        return _autodiff_jax
    if root == "tensorflow":
        from . import _autodiff_tf

        return _autodiff_tf
    return None


def _plain(v: np.ndarray) -> Any:
    return v.item() if v.ndim == 0 else v


def _tracked(x: Any):
    """Return (framework tensor, rebuild) for a tracked argument, or None.

    ``rebuild`` turns NumPy values of the tensor back into an argument.
    """
    values = getattr(x, "_tvalues", None)
    if values is not None:
        return values, x._with_values
    x = _unwrap(x)
    if _framework(x) is not None:
        return x, _plain
    return None


def _content(times: np.ndarray, values: Any, adapter: Any, dtype: Any) -> np.ndarray:
    """(n, 2) breakpoints with the framework `values`, or NaN while they are abstract."""
    concrete = adapter.concrete(values)
    if concrete is None:
        concrete = np.full(len(times), np.nan)
    return np.column_stack([times, concrete]).astype(dtype)


def content_from_values(times: Any, values: Any) -> tuple[np.ndarray, Any]:
    """(n, 2) breakpoints from separate times and values, and the framework
    values if `values` is tracked, otherwise None."""
    times = np.asarray(times)
    tracked = _tracked(values)
    tensor = tracked[0] if tracked is not None else np.asarray(values)
    if times.ndim != 1 or tuple(tensor.shape) != times.shape:
        raise ValueError(
            "values and times must be 1-D of the same length "
            f"(got shapes {tuple(tensor.shape)} and {times.shape})."
        )
    if tracked is None:
        return np.column_stack([times, tensor]), None
    dtype = np.float32 if times.dtype == np.float32 else np.float64
    return _content(times, tensor, _framework(tensor), dtype), tensor


def pcf_op(primal_op: Callable[..., Any], combine: Callable[..., Any], *operands: Any) -> Any:
    """An elementwise operation on PCFs, tensors of PCFs, and scalars.

    Untracked, this is ``primal_op(*operands)``. If any operand is tracked, the
    result has, for each element, the union of the operands' breakpoint times
    (with operands broadcast against each other), and framework values
    ``combine`` of the operands' values at those times. The union does not
    depend on the values, so this works while a framework traces abstract
    values, as JAX does. The PCFs in the result hold the values when they are
    concrete and NaN while they are abstract.
    """
    tracked = [t for t in map(_tracked, operands) if t is not None]
    if not tracked:
        return primal_op(*operands)
    like = tracked[0][0]
    adapter = _framework(like)

    pcf_positions = [i for i, op in enumerate(operands) if hasattr(op, "_serial")]
    serials = [operands[i]._serial() for i in pcf_positions]
    shapes = [enumeration.shape[:-1] for _, enumeration in serials]
    shape = np.broadcast_shapes(*shapes)
    owners = [np.broadcast_to(np.arange(int(np.prod(s))).reshape(s), shape).ravel() for s in shapes]
    enumerations = [enumeration.reshape(-1, 2) for _, enumeration in serials]

    # ponytail: per-element loop in Python; move to C++ if it shows up in profiles
    times, gathers = [], [[] for _ in serials]
    for k in range(int(np.prod(shape))):
        ranges = [enumeration[owner[k]] for enumeration, owner in zip(enumerations, owners)]
        t = np.unique(np.concatenate([c[a:b, 0] for (c, _), (a, b) in zip(serials, ranges)]))
        times.append(t)
        for gather, (c, _), (a, b) in zip(gathers, serials, ranges):
            gather.append(a + np.searchsorted(c[a:b, 0], t, side="right") - 1)

    def values(i, op):
        if i not in pcf_positions:
            return _unwrap(op)
        j = pcf_positions.index(i)
        v = op._tvalues
        if v is None:
            v = adapter.as_tensor(serials[j][0][:, 1], like=like)
        return adapter.gather(v, np.concatenate(gathers[j]))

    result_values = combine(*(values(i, op) for i, op in enumerate(operands)))

    sizes = np.array([len(t) for t in times])
    ends = np.cumsum(sizes)
    enumeration = np.stack([ends - sizes, ends], axis=-1).reshape(*shape, 2)
    content = _content(np.concatenate(times), result_values, adapter, serials[0][0].dtype)

    if shape == ():
        from .functional.pcf import Pcf

        result = Pcf(content)
    else:
        from .serialize import from_serial_content

        result = from_serial_content(content, enumeration)
    result._tvalues = result_values
    return result


def differentiable(
    *wrt: str, vjp: Callable[..., tuple], out: Callable[..., tuple] | None = None
) -> Callable[[F], F]:
    """``out``, given the function's arguments, returns the (shape, NumPy dtype)
    of its result. Frameworks that trace abstract values, such as JAX, need it."""
    def decorate(fn: F) -> F:
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            tracked = {}
            for name in wrt:
                t = _tracked(bound.arguments[name])
                if t is not None:
                    tracked[name] = t
            if not tracked:
                return fn(*args, **kwargs)
            names = list(tracked)
            tensors = [tracked[name][0] for name in names]

            def with_values(values):
                arguments = dict(bound.arguments)
                for name, v in zip(names, values):
                    arguments[name] = tracked[name][1](v)
                return arguments

            def primal(*values):
                return np.asarray(fn(**with_values(values)))

            def backward(grad_out, *values):
                grads = vjp(grad_out, **with_values(values))
                return tuple(grads[wrt.index(name)] for name in names)

            out_spec = out(**bound.arguments) if out is not None else None
            return _framework(tensors[0]).apply(primal, backward, *tensors, out=out_spec)

        return wrapper  # type: ignore[return-value]

    return decorate
