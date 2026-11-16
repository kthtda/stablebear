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


def pcf_parts(pcfs: Any, build: Callable[[np.ndarray, np.ndarray], Any] | None = None) -> _Tracked | None:
    """The tracked parts of a Pcf, PcfTensor, or Plf: its framework times and
    values. ``build`` makes the object from serial content and enumeration,
    by default a Pcf or PcfTensor."""
    columns = [(col, t) for col, t in ((0, pcfs._ttimes), (1, pcfs._tvalues)) if t is not None]
    if not columns:
        return None
    content, enumeration = pcfs._serial()

    def rebuild(arrays):
        c = content.copy()
        for (col, _), array in zip(columns, arrays):
            c[:, col] = array
        return (build or _pcfs_from_content)(c, enumeration)

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


def _union_of_times(
    pcfs: list, serials: list, enumerations: list, owners: list, offsets: np.ndarray, shape: tuple
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """The union of the breakpoint times of one or two operands, per result element.

    Operand j, ``pcfs[j]``, has serial content ``serials[j][0]`` and flat
    enumeration ``enumerations[j]``, and result element k, of a result of the
    given shape, takes its element ``owners[j][k]``. Returns the times of all
    result elements, concatenated; the number of times of each; for each
    time, the row of the breakpoint that gives it, of the first operand that
    has one there, counted over the contents of all operands with operand j
    starting at ``offsets[j]``; and per operand, for each time, the row of
    its breakpoint that holds at it.
    """
    from . import _sb_cpp as cpp

    if len(pcfs) == 1:
        content, enumeration = serials[0][0], enumerations[0]
        rows = np.arange(len(content), dtype=np.int64)
        return content[:, 0], enumeration[:, 1] - enumeration[:, 0], rows, [rows]

    def tensor(op, content, enumeration):
        # A single Pcf as a 0-d tensor.
        if enumeration.ndim == 1:
            data = _pcfs_tensor(content, enumeration)
        else:
            data = op._cpp() if hasattr(op, "_cpp") else op._data
        return data.broadcast_to(list(shape))

    times, sizes, f_local, g_local, from_g = cpp.merge_breakpoints(
        *(tensor(op, content, enumeration) for op, (content, enumeration) in zip(pcfs, serials)))
    element = np.repeat(np.arange(len(sizes)), sizes)
    f_rows = enumerations[0][owners[0][element], 0] + f_local
    g_rows = enumerations[1][owners[1][element], 0] + g_local
    return times, sizes, np.where(from_g, offsets[1] + g_rows, f_rows), [f_rows, g_rows]


def _pcfs_tensor(content: np.ndarray, enumeration: np.ndarray) -> Any:
    """The C++ 0-d tensor holding the one PCF with the given serial content."""
    from .serialize import from_serial_content

    return from_serial_content(content, enumeration[None])._data.reshape([])


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

    times, sizes, sources, gathers = _union_of_times(pcfs, serials, enumerations, owners, offsets, shape)

    def values(i, op):
        if i not in pcf_positions:
            return _unwrap(op)
        j = pcf_positions.index(i)
        v = op._tvalues
        if v is None:
            v = adapter.as_tensor(serials[j][0][:, 1], like=like)
        return adapter.gather(v, gathers[j])

    result_values = combine(*(values(i, op) for i, op in enumerate(operands)))
    result_times = None
    if any(op._ttimes is not None for op in pcfs):
        all_times = adapter.concat([
            op._ttimes if op._ttimes is not None else adapter.as_tensor(c[:, 0], like=like)
            for op, (c, _) in zip(pcfs, serials)
        ])
        result_times = adapter.gather(all_times, sources)

    ends = np.cumsum(sizes)
    enumeration = np.stack([ends - sizes, ends], axis=-1).reshape(*shape, 2)
    dtype = serials[0][0].dtype
    if enumeration.ndim > 1:
        from .base_tensor import _TrackedPcfTensor

        return _TrackedPcfTensor(
            [result_times if result_times is not None else times.astype(dtype), result_values], enumeration)
    content = np.column_stack([times, _concrete_column(result_values, len(times), adapter)])
    result = _pcfs_from_content(content.astype(dtype), enumeration)
    result._tvalues = result_values
    result._ttimes = result_times
    return result


def differentiable(
    *wrt: str,
    vjp: Callable[..., tuple],
    out: Callable[..., tuple] | None = None,
    residuals: Callable[..., list] | None = None,
) -> Callable[[F], F]:
    """``out``, given the function's arguments, returns the (shape, NumPy dtype)
    of its result. Frameworks that trace abstract values, such as JAX, need it.

    With ``residuals``, the function returns ``(result, residuals)``, where
    the residuals are a tuple of arrays from its computation that the vjp
    reuses: the vjp receives them as its ``residuals`` argument, so the
    backward pass need not repeat the forward pass. ``residuals``, given the
    function's arguments, returns their (shape, NumPy dtype)s."""
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
                result = fn(*args, **kwargs)
                return result[0] if residuals is not None else result
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
                result = fn(**with_values(values))
                if residuals is None:
                    return np.asarray(result), ()
                return np.asarray(result[0]), tuple(result[1])

            def backward(grad_out, saved, *values):
                extra = {"residuals": saved} if residuals is not None else {}
                grads = vjp(grad_out, **with_values(values), **extra)
                return tuple(g for name in names for g in tracked[name].split(grads[wrt.index(name)]))

            out_spec = out(**bound.arguments) if out is not None else None
            residual_specs = residuals(**bound.arguments) if residuals is not None else []
            return _framework(tensors[0]).apply(
                primal, backward, *tensors, out=out_spec, residuals=residual_specs)

        return wrapper  # type: ignore[return-value]

    return decorate
