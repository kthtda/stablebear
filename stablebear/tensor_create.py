import operator
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, overload

from . import _sb_cpp as cpp
from ._tensor_base import Shape, ShapeLike, Tensor
from .base_tensor import (
    BoolTensor,
    FloatTensor,
    IntPcfTensor,
    IntTensor,
    PcfTensor,
    _first_pcf,
    _to_tensor_pcf,
)
from .nested_tensor import NestedTensor
from .point_cloud import PointCloudTensor
from .typing import (
    _NATIVE_SUFFIX,
    Dtype,
    _assert_valid_dtype,
    _SeriesDtype,
    barcode32,
    barcode64,
    boolean,
    distmat32,
    distmat64,
    float32,
    float64,
    int32,
    int64,
    pcf32,
    pcf32i,
    pcf64,
    pcf64i,
    pcloud32,
    pcloud64,
    symmat32,
    symmat64,
    uint32,
    uint64,
)

if TYPE_CHECKING:
    from .timeseries.series import TimeSeries

cpp_p = cpp.persistence


# Single source of truth for dtype dispatch, shared by zeros() and tensor().
# Maps each dtype sentinel to (wrapper class, C++ tensor class, zero-fill args).
# Built lazily and cached because the matrix / barcode wrappers import back
# into this module.
_DTYPE_DISPATCH = {}

# Families whose constructor selects precision via a ``dtype=`` keyword; the
# rest (bool, PCF, barcode) infer it from the data or have a single precision.
_DATA_DTYPE_KW = frozenset({
    float32, float64, int32, int64, uint32, uint64,
    pcloud32, pcloud64, distmat32, distmat64, symmat32, symmat64,
})


def _dtype_dispatch():
    if not _DTYPE_DISPATCH:
        from .persistence.ph_tensor import BarcodeTensor
        from .distance_matrix import DistanceMatrixTensor
        from .symmetric_matrix import SymmetricMatrixTensor
        # Insertion order is the canonical dtype list used in error messages.
        _DTYPE_DISPATCH.update({
            pcf32: (PcfTensor, cpp.Pcf32Tensor, ()),
            pcf64: (PcfTensor, cpp.Pcf64Tensor, ()),
            pcf32i: (IntPcfTensor, cpp.Pcf32iTensor, ()),
            pcf64i: (IntPcfTensor, cpp.Pcf64iTensor, ()),
            float32: (FloatTensor, cpp.Float32Tensor, (0.0,)),
            float64: (FloatTensor, cpp.Float64Tensor, (0.0,)),
            int32: (IntTensor, cpp.Int32Tensor, (0,)),
            int64: (IntTensor, cpp.Int64Tensor, (0,)),
            uint32: (IntTensor, cpp.Uint32Tensor, (0,)),
            uint64: (IntTensor, cpp.Uint64Tensor, (0,)),
            boolean: (BoolTensor, cpp.BoolTensor, (False,)),
            pcloud32: (PointCloudTensor, cpp.PointCloud32Tensor, ()),
            pcloud64: (PointCloudTensor, cpp.PointCloud64Tensor, ()),
            barcode32: (BarcodeTensor, cpp_p.Barcode32Tensor, ()),
            barcode64: (BarcodeTensor, cpp_p.Barcode64Tensor, ()),
            distmat32: (DistanceMatrixTensor, cpp.DistanceMatrix32Tensor, ()),
            distmat64: (DistanceMatrixTensor, cpp.DistanceMatrix64Tensor, ()),
            symmat32: (SymmetricMatrixTensor, cpp.SymmetricMatrix32Tensor, ()),
            symmat64: (SymmetricMatrixTensor, cpp.SymmetricMatrix64Tensor, ()),
        })
    return _DTYPE_DISPATCH


def zeros(shape: ShapeLike, dtype: Dtype = pcf32) -> Any:  # Any: the type follows the dtype.
    """
    Creates a new `Tensor` of the specified `shape` and `dtype` whose entries are "zero." What "zero" means depends on the `dtype`:

    `dtype=pcf32/64`: A PCF that takes the value 0 for all times.
    `dtype=pcf32i/64i`: An integer PCF that takes the value 0 for all times.
    `dtype=float32/float64`: The number 0.
    `dtype=pcloud32/64`: An empty point cloud.
    `dtype=barcode32/64`: An empty barcode.
    `dtype=symmat32/64`: A 0×0 symmetric matrix.
    `dtype=distmat32/64`: A 0×0 distance matrix.
    `dtype=ts(value_type)`: A series of one channel without observations,
    in seconds, to be replaced by assigning a `TimeSeries` to the element.
    Evaluating a series without observations raises `ValueError`.

    Parameters
    ----------
    shape : ShapeLike
        Shape of the returned tensor
    dtype
        The data type of the elements

    Returns
    -------
    Tensor
        The newly created tensor
    """

    if not isinstance(shape, Shape):
        shape = Shape(shape)  # If passed as, e.g., tuple of ints

    if isinstance(dtype, _SeriesDtype):
        from .timeseries.series import TimeSeriesTensor
        return TimeSeriesTensor(getattr(cpp, f"TimeSeries{_NATIVE_SUFFIX[dtype.value_type]}Tensor")(shape))

    dispatch = _dtype_dispatch()
    _assert_valid_dtype(dtype, list(dispatch))
    wrapper, cpp_cls, zero_fill = dispatch[dtype]
    return wrapper(cpp_cls(shape, *zero_fill))


def tensor(data, dtype: Dtype | None = None) -> Any:  # Any: the type follows the data and dtype.
    """Create a tensor from existing data, dispatching on *dtype*.

    A NumPy-like factory that wraps the per-family constructors so any
    supported tensor can be built from an ndarray (or nested list) in one
    call, instead of allocating with :func:`zeros` and assigning elements in
    a Python loop.

    Parameters
    ----------
    data : ndarray, list, or tuple
        The source data. For point-cloud / matrix / barcode tensors this is
        interpreted by the corresponding constructor (see those classes).
    dtype : Dtype, optional
        Target element dtype. When ``None``, a numeric dtype is inferred from
        the array (bool/int/float), while a collection of ``Pcf`` objects is
        inferred as a ``PcfTensor`` or ``IntPcfTensor``, and one of
        :class:`~stablebear.timeseries.TimeSeries` as a
        :class:`~stablebear.timeseries.TimeSeriesTensor` of dtype
        ``ts(value_type)``. The series may differ in their times, units,
        and channels, but not in the dtype of their observations. Other
        non-numeric tensors require an explicit dtype.

    Raises
    ------
    TypeError
        If series differ in the dtype of their observations, or do not
        match an explicit ``dtype``.

    Returns
    -------
    Tensor
    """
    def contains_tensor(value):
        if isinstance(value, Tensor):
            return True
        return isinstance(value, (list, tuple)) and any(
            contains_tensor(child) for child in value
        )

    if isinstance(data, (list, tuple)) and contains_tensor(data):
        return NestedTensor(data, dtype=dtype)

    from .timeseries.series import TimeSeries, TimeSeriesTensor
    if isinstance(dtype, _SeriesDtype) or (dtype is None and isinstance(_first_leaf(data), TimeSeries)):
        return TimeSeriesTensor._from_series(data, dtype)

    if dtype is None:
        if _first_pcf(data) is not None:
            return _to_tensor_pcf(data)

        import numpy as np
        arr = np.asarray(data)
        if arr.dtype == np.bool_:
            return BoolTensor(arr)
        if np.issubdtype(arr.dtype, np.integer):
            return IntTensor(arr)
        if np.issubdtype(arr.dtype, np.floating):
            return FloatTensor(arr)
        raise TypeError(
            f"Cannot infer a tensor dtype from data of dtype {arr.dtype}; "
            "pass an explicit dtype= argument")

    dispatch = _dtype_dispatch()
    # Barcode tensors are not constructible from raw array/list data.
    if dtype not in dispatch or dtype in (barcode32, barcode64):
        raise TypeError(f"Unsupported dtype {dtype} for tensor()")
    wrapper = dispatch[dtype][0]
    if dtype in _DATA_DTYPE_KW:
        return wrapper(data, dtype=dtype)
    return wrapper(data)


def _first_leaf(data):
    """The first element of nested lists or tuples, or ``None`` if empty."""
    while isinstance(data, (list, tuple)):
        if not data:
            return None
        data = data[0]
    return data


def indices(data):
    """Create a ``uint64`` tensor for use as indices.

    This is shorthand for ``tensor(data, dtype=uint64)``.
    """
    return tensor(data, dtype=uint64)


def _series_channels(tensors):
    """The series whose channels a join combines, or ``None`` for tensors."""
    from .timeseries.series import TimeSeries
    if isinstance(tensors, TimeSeries):
        return tensors
    if any(isinstance(t, TimeSeries) for t in tensors):
        raise TypeError(
            "Pass one TimeSeries to combine its channels; build it from several "
            "with TimeSeries.from_channels, or make a tensor of series with sb.tensor"
        )
    return None


@overload
def concatenate(tensors: "TimeSeries", axis: int = 0) -> "TimeSeries": ...
@overload
def concatenate(tensors: Sequence[Any], axis: int = 0) -> Any: ...
def concatenate(tensors, axis=0):
    """Concatenate tensors, or the observations of time series, along an
    existing axis.

    Tensors are joined along their axis ``axis`` (outer indexing): tensors
    of shapes ``(2, 4)`` and ``(3, 4)`` give one of shape ``(5, 4)``.

    For time series, the result is a series of one channel whose
    observation at each time joins the channels' observations in effect
    there along their axis ``axis``: channels of shapes ``(2,)`` and
    ``(3,)`` give observations of shape ``(5,)``.
    Its times are the start of the domain and every time in the domain at
    which a channel stores an observation. The domain, unit, epoch, and
    origin stay the same, and the snap tolerance is the smallest of the
    channels'. Observations are copied.

    .. warning::

       Observations can be dropped or repeated. A channel's observations
       outside the domain, or at an end the domain leaves out, are
       dropped, and an observation in effect at several of the times is
       repeated at each.

    Parameters
    ----------
    tensors : sequence of Tensor, or TimeSeries
        Tensors of one type, whose shapes match except along ``axis``. A
        ``TimeSeries`` joins its channels; make several series the
        channels of one with
        :meth:`~stablebear.timeseries.TimeSeries.from_channels` first.
        To put separate series into a tensor, use :func:`tensor`.
    axis : int, default 0
        The axis to join along, of each tensor or of each observation; a
        negative ``axis`` counts from the end.

    Returns
    -------
    Tensor or TimeSeries
        A tensor of the type of the inputs, or a series of one channel.

    Raises
    ------
    ValueError
        If there is nothing to join, the shapes differ other than along
        ``axis``, or ``axis`` is out of range. For series, also if the
        observations are scalars, the channels have different origins,
        their domains do not overlap, or a channel has no observations.
    TypeError
        If a sequence holds a ``TimeSeries``.
    """
    series = _series_channels(tensors)
    if series is not None:
        return series._from_cpp(cpp.concatenate_channels(series._data, operator.index(axis)))
    if not tensors:
        raise ValueError("need at least one tensor to concatenate")
    _validate_nested_join(tensors)
    cpp_tensors = [t._data for t in tensors]
    result = type(cpp_tensors[0]).concatenate(cpp_tensors, axis)
    return tensors[0]._to_py_tensor(result)


@overload
def stack(tensors: "TimeSeries", axis: int = 0) -> "TimeSeries": ...
@overload
def stack(tensors: Sequence[Any], axis: int = 0) -> Any: ...
def stack(tensors, axis=0):
    """Stack tensors, or the observations of time series, along a new axis.

    ``n`` tensors of shape ``shape`` give one of shape ``(n, *shape)`` for
    ``axis=0``.

    For time series, the result is a series of one channel whose
    observation at each time stacks the channels' observations in effect
    there along a new axis ``axis`` of length ``n_channels``.
    Its times are the start of the domain and every time in the domain at
    which a channel stores an observation. The domain, unit, epoch, and
    origin stay the same, and the snap tolerance is the smallest of the
    channels'. Observations are copied.

    .. warning::

       Observations can be dropped or repeated. A channel's observations
       outside the domain, or at an end the domain leaves out, are
       dropped, and an observation in effect at several of the times is
       repeated at each.

    Parameters
    ----------
    tensors : sequence of Tensor, or TimeSeries
        Tensors of one type and shape. A ``TimeSeries`` stacks its
        channels; make several series the channels of one with
        :meth:`~stablebear.timeseries.TimeSeries.from_channels` first.
        To put separate series into a tensor, use :func:`tensor`.
    axis : int, default 0
        The position of the new axis, in the result or in each
        observation; a negative ``axis`` counts from the end.

    Returns
    -------
    Tensor or TimeSeries
        A tensor of the type of the inputs, or a series of one channel.

    Raises
    ------
    ValueError
        If there is nothing to stack, the shapes differ, or ``axis`` is
        out of range. For series, also if the channels have different
        origins, their domains do not overlap, or a channel has no
        observations.
    TypeError
        If a sequence holds a ``TimeSeries``.
    """
    series = _series_channels(tensors)
    if series is not None:
        return series._from_cpp(cpp.stack_channels(series._data, operator.index(axis)))
    if not tensors:
        raise ValueError("need at least one tensor to stack")
    _validate_nested_join(tensors)
    cpp_tensors = [t._data for t in tensors]
    result = type(cpp_tensors[0]).stack(cpp_tensors, axis)
    return tensors[0]._to_py_tensor(result)


def _validate_nested_join(tensors):
    first = tensors[0]
    if not isinstance(first, NestedTensor):
        return
    for tensor in tensors[1:]:
        if not isinstance(tensor, NestedTensor):
            raise TypeError("all tensors in a nested join must be NestedTensor objects")
        if tensor.dtype is not first.dtype:
            raise TypeError(
                f"NestedTensor leaf dtype must be {first.dtype}, got {tensor.dtype}"
            )
        if tensor.depth != first.depth:
            raise ValueError(
                "all NestedTensor operands must have the same depth; "
                f"got {first.depth} and {tensor.depth}"
            )


def split(tensor, indices_or_sections, axis=0):
    """Split a tensor into sub-tensors along an axis.

    Parameters
    ----------
    tensor : Tensor
        The tensor to split.
    indices_or_sections : int or list of int
        If an int, the axis length must be evenly divisible by that number.
        The result contains that many contiguous parts, each of length
        ``axis_size // sections``. Use ``array_split`` when equal division is
        not possible.
        If a list ``[i0, i1, ...]``, its entries are boundaries and the
        resulting parts cover the half-open intervals ``[0, i0)``,
        ``[i0, i1)``, ..., ``[ik, axis_size)`` along *axis*. For example,
        splitting an axis of length 8 at ``[3, 5]`` produces parts of lengths
        3, 2, and 3. The element at each boundary starts the following part.
    axis : int
        The axis along which to split (default 0). A negative axis counts
        from the end.

    Returns
    -------
    list of Tensor
        A list of tensor views sharing data with the original.

    See Also
    --------
    array_split : Split allowing uneven divisions.
    """
    cpp_data = tensor._data
    if isinstance(indices_or_sections, int):
        parts = type(cpp_data).split_sections(cpp_data, indices_or_sections, axis)
    else:
        parts = type(cpp_data).split_indices(cpp_data, list(indices_or_sections), axis)
    return [tensor._to_py_tensor(p) for p in parts]


def array_split(tensor, indices_or_sections, axis=0):
    """Split a tensor into sub-tensors, allowing uneven splits.

    Parameters
    ----------
    tensor : Tensor
        The tensor to split.
    indices_or_sections : int or list of int
        If an int, the tensor is split into that many parts. Let
        ``axis_size = q * sections + r``. The first ``r`` parts have length
        ``q + 1``; the remaining trailing parts have length ``q`` and do not
        receive the extra element. For example, an axis of length 8 split into
        3 parts produces lengths 3, 3, and 2.
        If a list ``[i0, i1, ...]``, the parts use the same half-open intervals
        as ``split``: ``[0, i0)``, ``[i0, i1)``, ...,
        ``[ik, axis_size)``. The element at each boundary starts the following
        part.
    axis : int
        The axis along which to split (default 0). A negative axis counts
        from the end.

    Returns
    -------
    list of Tensor
        A list of tensor views sharing data with the original.

    See Also
    --------
    split : Split requiring equal divisions.
    """
    cpp_data = tensor._data
    if isinstance(indices_or_sections, int):
        parts = type(cpp_data).array_split(cpp_data, indices_or_sections, axis)
    else:
        parts = type(cpp_data).split_indices(cpp_data, list(indices_or_sections), axis)
    return [tensor._to_py_tensor(p) for p in parts]
