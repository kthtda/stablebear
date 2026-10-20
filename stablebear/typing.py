import numpy as np


class dtype:
    """Describes the element type of a stablebear tensor.

    Each dtype is a singleton instance (e.g. ``stablebear.pcf32``,
    ``stablebear.float64``). Use ``isinstance(x, stablebear.dtype)`` to
    check whether a value is a stablebear dtype.
    """

    def __init__(self, name: str, doc: str = ""):
        self._name = name
        self.__doc__ = doc

    @property
    def name(self) -> str:
        """Short name of this dtype (e.g. ``'pcf32'``)."""
        return self._name

    def __repr__(self):
        return f"stablebear.{self._name}"

    def __str__(self):
        return self._name

    # For backward compat with code that used dtype.__name__
    @property
    def __name__(self):
        return self._name

    def __hash__(self):
        return id(self)

    def __reduce__(self):
        # Pickle support: reconstruct by looking up the module-level name
        return self._name


# Scalar dtypes
float32 = dtype("float32", "32-bit floating-point scalar dtype.")
float64 = dtype("float64", "64-bit floating-point scalar dtype.")
int32 = dtype("int32", "32-bit integer scalar dtype.")
int64 = dtype("int64", "64-bit integer scalar dtype.")
uint32 = dtype("uint32", "32-bit unsigned integer scalar dtype.")
uint64 = dtype("uint64", "64-bit unsigned integer scalar dtype.")

# PCF dtypes
pcf32 = dtype("pcf32", "32-bit piecewise constant function dtype.")
pcf64 = dtype("pcf64", "64-bit piecewise constant function dtype.")
pcf32i = dtype("pcf32i", "32-bit integer piecewise constant function dtype.")
pcf64i = dtype("pcf64i", "64-bit integer piecewise constant function dtype.")

# Point cloud dtypes
pcloud32 = dtype("pcloud32", "32-bit point cloud dtype.")
pcloud64 = dtype("pcloud64", "64-bit point cloud dtype.")

# Barcode dtypes
barcode32 = dtype("barcode32", "32-bit persistence barcode dtype.")
barcode64 = dtype("barcode64", "64-bit persistence barcode dtype.")

# Matrix dtypes
symmat32 = dtype("symmat32", "32-bit symmetric matrix dtype.")
symmat64 = dtype("symmat64", "64-bit symmetric matrix dtype.")
distmat32 = dtype("distmat32", "32-bit distance matrix dtype.")
distmat64 = dtype("distmat64", "64-bit distance matrix dtype.")

# Boolean
boolean = dtype("boolean", "Boolean dtype.")


# Each tensor element dtype and the suffix of its native class names, such
# as cpp.TimeSeriesFloat64.
_NATIVE_SUFFIX = {
    boolean: "Bool",
    float32: "Float32", float64: "Float64",
    int32: "Int32", int64: "Int64",
    uint32: "Uint32", uint64: "Uint64",
    pcf32: "Pcf32", pcf64: "Pcf64",
    pcf32i: "Pcf32i", pcf64i: "Pcf64i",
    pcloud32: "PointCloud32", pcloud64: "PointCloud64",
    distmat32: "DistanceMatrix32", distmat64: "DistanceMatrix64",
    symmat32: "SymmetricMatrix32", symmat64: "SymmetricMatrix64",
    barcode32: "Barcode32", barcode64: "Barcode64",
}


class _SeriesDtype(dtype):
    def __init__(self, inner: dtype):
        super().__init__(
            f"series({inner.name})",
            f"Dtype of time series whose observations have dtype {inner.name}.",
        )
        self._inner = inner

    @property
    def inner(self) -> dtype:
        """The dtype of the observations of each series."""
        return self._inner

    def __repr__(self):
        return f"stablebear.series(stablebear.{self._inner.name})"

    def __reduce__(self):
        return series, (self._inner,)


_SERIES_DTYPES = {}


def series(inner: dtype) -> dtype:
    """The dtype of time series whose observations have dtype ``inner``.

    It is the dtype of a series of series, such as the windows of a series
    of ``inner`` observations. The same ``inner`` always gives the same
    object, so dtypes compare with ``is``, and :attr:`inner` reads it back.

    Raises
    ------
    TypeError
        If ``inner`` is not the dtype of a tensor element, such as
        ``float64`` or ``pcloud32``.
    """
    if inner not in _NATIVE_SUFFIX:
        raise TypeError(f"Series hold tensor elements, not {inner!r}")
    if inner not in _SERIES_DTYPES:
        _SERIES_DTYPES[inner] = _SeriesDtype(inner)
    return _SERIES_DTYPES[inner]


Dtype = dtype  # Alias for type annotations


_SB_TO_NP = {
    float32: np.float32,
    float64: np.float64,
    int32: np.int32,
    int64: np.int64,
    uint32: np.uint32,
    uint64: np.uint64,
}

_NP_TO_SB = {
    np.float32: float32,
    np.float64: float64,
    np.int32: int32,
    np.int64: int64,
    np.uint32: uint32,
    np.uint64: uint64,
}


def _assert_valid_dtype(dt, valid_dtypes):
    if not any(dt is valid_dtype for valid_dtype in valid_dtypes):
        raise TypeError(
            "Only the following dtypes are supported: "
            + ", ".join(str(d) for d in valid_dtypes)
            + f" (supplied {dt})"
        )


def _validate_dtype(dt, valid_dtypes):
    _assert_valid_dtype(dt, valid_dtypes)
    return dt


# Short tags used to look up C++ cast functions by naming convention:
# cpp.cast_{src_tag}_{dst_tag}
_DTYPE_TAG = {
    float32: "f32", float64: "f64",
    int32: "i32", int64: "i64",
    uint32: "u32", uint64: "u64",
    pcf32: "pcf32", pcf64: "pcf64",
    pcf32i: "pcf32i", pcf64i: "pcf64i",
    pcloud32: "pcloud32", pcloud64: "pcloud64",
}


# Populated lazily to avoid circular imports (tensor.py -> typing.py)
_DTYPE_TO_WRAPPER = {}


def _init_dtype_wrappers():
    if _DTYPE_TO_WRAPPER:
        return
    from .base_tensor import FloatTensor, IntTensor, PcfTensor, IntPcfTensor
    from .point_cloud import PointCloudTensor
    _DTYPE_TO_WRAPPER.update({
        float32: FloatTensor, float64: FloatTensor,
        int32: IntTensor, int64: IntTensor,
        uint32: IntTensor, uint64: IntTensor,
        pcf32: PcfTensor, pcf64: PcfTensor,
        pcf32i: IntPcfTensor, pcf64i: IntPcfTensor,
        pcloud32: PointCloudTensor, pcloud64: PointCloudTensor,
    })
