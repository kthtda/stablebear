"""Gradients of stablebear operations in other autodiff frameworks.

An operation opts in with the ``differentiable`` decorator, naming the
parameters that can carry gradients and a vector-Jacobian product:

    def _f_vjp(grad_out, x, p, verbose=False):
        return (d_x, d_p)   # gradient of sum(grad_out * f(x, p)) per name in wrt

    @differentiable("x", "p", vjp=_f_vjp)
    def f(x, p, verbose=False): ...

The function itself stays framework-free. A parameter is tracked when it is a
tensor of a supported framework, or a stablebear object that carries some, such
as a Pcf whose times or values depend on a framework tensor. The decorator
converts the tracked parameters to NumPy (rebuilding stablebear objects from
their parts), calls the function, and runs the call as an operation of that
framework, so the result carries gradients there. The vjp receives the same
arguments as the function and returns NumPy values: for a Pcf or PcfTensor an
(m, 2) array of gradients with respect to (time, value) of each breakpoint, in
the order of _serial. Entries for untracked parameters are ignored.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, NamedTuple, TypeVar

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


class _Tracked(NamedTuple):
    """The framework tensors behind a tracked argument.

    ``rebuild`` turns NumPy values of the tensors back into an argument, and
    ``split`` turns the vjp's gradient for the argument into one per tensor.
    """

    tensors: list
    rebuild: Callable[[list], Any]
    split: Callable[[Any], list]


def _tracked(x: Any) -> _Tracked | None:
    """The framework tensors behind `x`, or None if it is not tracked."""
    parts = getattr(x, "_tracked_parts", None)
    if parts is not None:
        return parts()
    x = _unwrap(x)
    if _framework(x) is not None:
        return _Tracked([x], lambda arrays: _plain(arrays[0]), lambda grad: [grad])
    return None


def _pcfs_from_content(content: np.ndarray, enumeration: np.ndarray) -> Any:
    """A Pcf (for a (2,) enumeration) or PcfTensor from serial content,
    without requiring strictly increasing times."""
    from .serialize import from_serial_content

    if enumeration.ndim == 1:
        return from_serial_content(content, enumeration[None])[0]
    return from_serial_content(content, enumeration)


def pcf_parts(pcfs: Any) -> _Tracked | None:
    """The tracked parts of a Pcf or PcfTensor: its framework times and values."""
    columns = [(col, t) for col, t in ((0, pcfs._ttimes), (1, pcfs._tvalues)) if t is not None]
    if not columns:
        return None
    content, enumeration = pcfs._serial()

    def rebuild(arrays):
        c = content.copy()
        for (col, _), array in zip(columns, arrays):
            c[:, col] = array
        return _pcfs_from_content(c, enumeration)

    def split(grad):
        return [np.asarray(grad)[:, col] for col, _ in columns]

    return _Tracked([t for _, t in columns], rebuild, split)


def _concrete_column(values: Any, n: int, adapter: Any) -> np.ndarray:
    concrete = adapter.concrete(values) if values is not None else None
    return concrete if concrete is not None else np.full(n, np.nan)


def content_from_values(times: Any, values: Any) -> tuple[np.ndarray, Any, Any]:
    """(n, 2) breakpoints from separate times and values, and the framework
    times and values where they are tracked (else None). Abstract times or
    values give NaN breakpoints.
    """
    times_tracked, values_tracked = _tracked(times), _tracked(values)
    t = times_tracked.tensors[0] if times_tracked else np.asarray(times)
    v = values_tracked.tensors[0] if values_tracked else np.asarray(values)
    if len(t.shape) != 1 or tuple(t.shape) != tuple(v.shape):
        raise ValueError(
            "values and times must be 1-D of the same length "
            f"(got shapes {tuple(v.shape)} and {tuple(t.shape)})."
        )
    if not times_tracked and not values_tracked:
        return np.column_stack([t, v]), None, None
    adapter = _framework((t if times_tracked else v))
    concrete_times = _concrete_column(t, t.shape[0], adapter) if times_tracked else t
    dtype = np.float32 if str(t.dtype).endswith("float32") else np.float64
    content = np.column_stack([concrete_times, _concrete_column(v, len(concrete_times), adapter)
                               if values_tracked else v]).astype(dtype)
    return content, t if times_tracked else None, v if values_tracked else None


_ABSTRACT_TIMES = (
    "PCF breakpoint times that JAX or tf.function traces abstractly are not supported here: "
    "the order of the breakpoints depends on their values."
)


def pcf_op(primal_op: Callable[..., Any], combine: Callable[..., Any], *operands: Any) -> Any:
    """An elementwise operation on PCFs, tensors of PCFs, and scalars.

    Untracked, this is ``primal_op(*operands)``. If any operand is tracked, the
    result has, for each element, the union of the operands' breakpoint times
    (with operands broadcast against each other), and framework values
    ``combine`` of the operands' values at those times. Each time of the result
    is the time of the first operand that has it, so gradients with respect to
    tracked times flow there. The union depends only on the times, so while a
    framework traces abstract values, as JAX does, tracked values work but
    tracked times do not. The PCFs in the result hold the values when they are
    concrete and NaN while they are abstract.
    """
    tracked = [t for t in map(_tracked, operands) if t is not None]
    if not tracked:
        return primal_op(*operands)
    like = tracked[0].tensors[0]
    adapter = _framework(like)

    pcf_positions = [i for i, op in enumerate(operands) if hasattr(op, "_serial")]
    pcfs = [operands[i] for i in pcf_positions]
    if any(op._ttimes is not None and adapter.concrete(op._ttimes) is None for op in pcfs):
        raise NotImplementedError(_ABSTRACT_TIMES)
    serials = [op._serial() for op in pcfs]
    offsets = np.cumsum([0] + [len(c) for c, _ in serials])
    shapes = [enumeration.shape[:-1] for _, enumeration in serials]
    shape = np.broadcast_shapes(*shapes)
    owners = [np.broadcast_to(np.arange(int(np.prod(s))).reshape(s), shape).ravel() for s in shapes]
    enumerations = [enumeration.reshape(-1, 2) for _, enumeration in serials]

    # ponytail: per-element loop in Python; move to C++ if it shows up in profiles
    times, sources, gathers = [], [], [[] for _ in serials]
    for k in range(int(np.prod(shape))):
        ranges = [enumeration[owner[k]] for enumeration, owner in zip(enumerations, owners)]
        candidates = np.concatenate([c[a:b, 0] for (c, _), (a, b) in zip(serials, ranges)])
        global_index = np.concatenate([offset + np.arange(a, b) for offset, (a, b) in zip(offsets, ranges)])
        t, first = np.unique(candidates, return_index=True)
        times.append(t)
        sources.append(global_index[first])
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
    result_times = None
    if any(op._ttimes is not None for op in pcfs):
        all_times = adapter.concat([
            op._ttimes if op._ttimes is not None else adapter.as_tensor(c[:, 0], like=like)
            for op, (c, _) in zip(pcfs, serials)
        ])
        result_times = adapter.gather(all_times, np.concatenate(sources))

    sizes = np.array([len(t) for t in times])
    ends = np.cumsum(sizes)
    enumeration = np.stack([ends - sizes, ends], axis=-1).reshape(*shape, 2)
    concat_times = np.concatenate(times)
    content = np.column_stack([concat_times, _concrete_column(result_values, len(concat_times), adapter)])
    result = _pcfs_from_content(content.astype(serials[0][0].dtype), enumeration)
    result._tvalues = result_values
    result._ttimes = result_times
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
            tensors = [tensor for name in names for tensor in tracked[name].tensors]
            ends = np.cumsum([len(tracked[name].tensors) for name in names])

            def with_values(values):
                arguments = dict(bound.arguments)
                starts = np.concatenate([[0], ends[:-1]])
                for name, start, end in zip(names, starts, ends):
                    arguments[name] = tracked[name].rebuild(list(values[start:end]))
                return arguments

            def primal(*values):
                return np.asarray(fn(**with_values(values)))

            def backward(grad_out, *values):
                grads = vjp(grad_out, **with_values(values))
                return tuple(g for name in names for g in tracked[name].split(grads[wrt.index(name)]))

            out_spec = out(**bound.arguments) if out is not None else None
            return _framework(tensors[0]).apply(primal, backward, *tensors, out=out_spec)

        return wrapper  # type: ignore[return-value]

    return decorate
