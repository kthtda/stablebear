from __future__ import annotations

import operator

import numpy as np

from . import _sb_cpp as cpp
from ._tensor_base import Tensor, _infer_shape_and_flatten
from .typing import (
    _NP_TO_SB,
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


_NESTED_CPP_TYPES = {
    float32: (cpp.NestedFloat32, cpp.NestedFloat32Tensor),
    float64: (cpp.NestedFloat64, cpp.NestedFloat64Tensor),
    int32: (cpp.NestedInt32, cpp.NestedInt32Tensor),
    int64: (cpp.NestedInt64, cpp.NestedInt64Tensor),
    uint32: (cpp.NestedUint32, cpp.NestedUint32Tensor),
    uint64: (cpp.NestedUint64, cpp.NestedUint64Tensor),
    boolean: (cpp.NestedBool, cpp.NestedBoolTensor),
    pcf32: (cpp.NestedPcf32, cpp.NestedPcf32Tensor),
    pcf64: (cpp.NestedPcf64, cpp.NestedPcf64Tensor),
    pcf32i: (cpp.NestedPcf32i, cpp.NestedPcf32iTensor),
    pcf64i: (cpp.NestedPcf64i, cpp.NestedPcf64iTensor),
    pcloud32: (cpp.NestedPointCloud32, cpp.NestedPointCloud32Tensor),
    pcloud64: (cpp.NestedPointCloud64, cpp.NestedPointCloud64Tensor),
    distmat32: (cpp.NestedDistanceMatrix32, cpp.NestedDistanceMatrix32Tensor),
    distmat64: (cpp.NestedDistanceMatrix64, cpp.NestedDistanceMatrix64Tensor),
    symmat32: (cpp.NestedSymmetricMatrix32, cpp.NestedSymmetricMatrix32Tensor),
    symmat64: (cpp.NestedSymmetricMatrix64, cpp.NestedSymmetricMatrix64Tensor),
    barcode32: (cpp.persistence.NestedBarcode32, cpp.persistence.NestedBarcode32Tensor),
    barcode64: (cpp.persistence.NestedBarcode64, cpp.persistence.NestedBarcode64Tensor),
}
_NESTED_NODE_TO_DTYPE = {
    node_type: dtype for dtype, (node_type, _) in _NESTED_CPP_TYPES.items()
}
_NESTED_TENSOR_TO_DTYPE = {
    tensor_type: dtype for dtype, (_, tensor_type) in _NESTED_CPP_TYPES.items()
}


def _leaf_wrappers():
    """Leaf dtype -> the tensor class that wraps a leaf of that dtype."""
    from .base_tensor import BoolTensor, FloatTensor, IntPcfTensor, IntTensor, PcfTensor
    from .distance_matrix import DistanceMatrixTensor
    from .persistence.ph_tensor import BarcodeTensor
    from .point_cloud import PointCloudTensor
    from .symmetric_matrix import SymmetricMatrixTensor

    return {
        float32: FloatTensor, float64: FloatTensor,
        int32: IntTensor, int64: IntTensor, uint32: IntTensor, uint64: IntTensor,
        boolean: BoolTensor,
        pcf32: PcfTensor, pcf64: PcfTensor,
        pcf32i: IntPcfTensor, pcf64i: IntPcfTensor,
        pcloud32: PointCloudTensor, pcloud64: PointCloudTensor,
        distmat32: DistanceMatrixTensor, distmat64: DistanceMatrixTensor,
        symmat32: SymmetricMatrixTensor, symmat64: SymmetricMatrixTensor,
        barcode32: BarcodeTensor, barcode64: BarcodeTensor,
    }


def _leaf_tensor(data, dtype=None):
    from ._tensor_base import IndexedElementTensor
    from .base_tensor import BoolTensor, FloatTensor, IntTensor

    wrappers = _leaf_wrappers()
    if isinstance(data, Tensor) and not isinstance(data, NestedTensor):
        if data.dtype not in wrappers or not isinstance(data, wrappers[data.dtype]):
            raise TypeError(f"Unsupported NestedTensor leaf {type(data).__name__}")
        if dtype is not None and data.dtype is not dtype:
            raise TypeError(
                f"NestedTensor leaf dtype must be {dtype}, got {data.dtype}"
            )
        leaf = data
        if isinstance(leaf, IndexedElementTensor) and leaf._has_indexed_storage():
            # A leaf holds ordinary storage, not a selection of another tensor.
            leaf = leaf.to_dense()
    elif isinstance(data, np.ndarray):
        if dtype is None:
            dtype = boolean if data.dtype == np.bool_ else _NP_TO_SB.get(data.dtype.type)
            if dtype is None:
                if np.issubdtype(data.dtype, np.floating):
                    dtype = float64
                elif np.issubdtype(data.dtype, np.signedinteger):
                    dtype = int64
                elif np.issubdtype(data.dtype, np.unsignedinteger):
                    dtype = uint64
        if dtype in (float32, float64):
            leaf = FloatTensor(data, dtype=dtype)
        elif dtype in (int32, int64, uint32, uint64):
            leaf = IntTensor(data, dtype=dtype)
        elif dtype is boolean:
            leaf = BoolTensor(data)
        else:
            raise TypeError(f"Unsupported NestedTensor leaf dtype {dtype}")
    else:
        raise TypeError(
            "NestedTensor leaves must be stablebear Tensor or NumPy array objects"
        )

    if leaf.dtype not in _NESTED_CPP_TYPES:
        raise TypeError(f"Unsupported NestedTensor leaf dtype {leaf.dtype}")
    return leaf


def _nested_node(value, dtype=None):
    if isinstance(value, NestedTensor):
        if dtype is not None and value.dtype is not dtype:
            raise TypeError(
                f"NestedTensor leaf dtype must be {dtype}, got {value.dtype}"
            )
        return value._root, value.dtype

    leaf = _leaf_tensor(value, dtype=dtype)
    node_type, _ = _NESTED_CPP_TYPES[leaf.dtype]
    return node_type._from_leaf_view(leaf._data), leaf.dtype


def _normalize_depth(depth):
    if depth is None:
        return None
    if isinstance(depth, (bool, np.bool_)):
        raise TypeError("NestedTensor depth must be an integer")
    try:
        depth = operator.index(depth)
    except TypeError as error:
        raise TypeError("NestedTensor depth must be an integer") from error
    if depth < 2:
        raise ValueError("Python NestedTensor depth must be at least 2")
    return depth


class NestedTensor(Tensor):
    """A recursively nested tensor whose leaf tensors share one dtype.

    Leaves may hold any element type, such as numbers, PCFs, point clouds,
    or barcodes; NumPy arrays become numeric or boolean leaves.

    Construction and assignment recursively copy supplied values. Outer views
    share storage; ``copy()`` and ``deepcopy()`` recursively isolate it.
    """

    def __init__(self, data, *, dtype=None, depth=None):
        from .base_tensor import FloatTensor, IntTensor

        super().__init__()
        depth = _normalize_depth(depth)
        if dtype is not None and dtype not in _NESTED_CPP_TYPES:
            raise TypeError(f"Unsupported NestedTensor leaf dtype {dtype}")

        if isinstance(data, NestedTensor):
            if dtype is not None and data.dtype is not dtype:
                raise TypeError(
                    f"NestedTensor leaf dtype must be {dtype}, got {data.dtype}"
                )
            dtype = data.dtype
            root = data._root.copy()
            if depth is not None and depth != root.depth:
                raise ValueError(
                    f"NestedTensor depth is {root.depth}, not requested depth {depth}"
                )
        elif isinstance(data, (FloatTensor, IntTensor, np.ndarray)):
            leaf = _leaf_tensor(data, dtype=dtype)
            dtype = leaf.dtype
            node_type, tensor_type = _NESTED_CPP_TYPES[dtype]
            tensor = tensor_type(cpp.Shape([]))
            tensor._set_element([], node_type._from_leaf_view(leaf._data))
            root = node_type._from_outer_view(tensor, 1)
            if depth is not None and depth != root.depth:
                raise ValueError(
                    f"NestedTensor depth is {root.depth}, not requested depth {depth}"
                )
        elif isinstance(data, (list, tuple)):
            shape, values = _infer_shape_and_flatten(data)
            if not values:
                if dtype is None:
                    raise TypeError(
                        "Empty NestedTensor requires an explicit leaf dtype"
                    )
                if depth is None:
                    raise ValueError(
                        "Empty NestedTensor requires an explicit depth"
                    )
                node_type, tensor_type = _NESTED_CPP_TYPES[dtype]
                children = tensor_type(cpp.Shape(list(shape or (0,))))
                root = node_type._from_outer_view(children, depth - 1)
            else:
                first, dtype = _nested_node(values[0], dtype=dtype)
                nodes = [first]
                nodes.extend(
                    _nested_node(value, dtype=dtype)[0]
                    for value in values[1:]
                )
                child_depth = nodes[0].depth
                if any(node.depth != child_depth for node in nodes[1:]):
                    raise ValueError(
                        "NestedTensor elements must have the same nesting depth"
                    )
                inferred_depth = child_depth + 1
                if depth is not None and depth != inferred_depth:
                    raise ValueError(
                        f"NestedTensor depth is {inferred_depth}, not requested depth {depth}"
                    )
                node_type, tensor_type = _NESTED_CPP_TYPES[dtype]
                tensor = tensor_type(cpp.Shape([len(nodes)]))
                for i, node in enumerate(nodes):
                    tensor._set_element([i], node)
                if shape != (len(nodes),):
                    tensor = tensor.reshape(list(shape))
                root = node_type._from_outer_view(tensor, child_depth)
        elif type(data) in _NESTED_NODE_TO_DTYPE:
            actual_dtype = _NESTED_NODE_TO_DTYPE[type(data)]
            if dtype is not None and actual_dtype is not dtype:
                raise TypeError(
                    f"NestedTensor leaf dtype must be {dtype}, got {actual_dtype}"
                )
            dtype = actual_dtype
            if data.is_leaf:
                raise TypeError("A Python NestedTensor must have a nested root")
            if depth is not None and depth != data.depth:
                raise ValueError(
                    f"NestedTensor depth is {data.depth}, not requested depth {depth}"
                )
            root = data.copy()
        elif type(data) in _NESTED_TENSOR_TO_DTYPE:
            actual_dtype = _NESTED_TENSOR_TO_DTYPE[type(data)]
            if dtype is not None and actual_dtype is not dtype:
                raise TypeError(
                    f"NestedTensor leaf dtype must be {dtype}, got {actual_dtype}"
                )
            dtype = actual_dtype
            node_type, _ = _NESTED_CPP_TYPES[dtype]
            child_depth = 0 if depth is None else depth - 1
            root = node_type(data, child_depth)
            if depth is not None and depth != root.depth:
                raise ValueError(
                    f"NestedTensor depth is {root.depth}, not requested depth {depth}"
                )
        else:
            raise TypeError(f"Cannot create NestedTensor from {type(data)}")

        self._root = root
        self._data = root.nested
        self.dtype = dtype

    @classmethod
    def _from_cpp(cls, data):
        """Wrap an internal node without copying its already-owned storage."""
        result = cls.__new__(cls)
        Tensor.__init__(result)
        result._root = data
        result._data = data.nested
        result.dtype = _NESTED_NODE_TO_DTYPE[type(data)]
        return result

    @classmethod
    def _is_cpp_nested_tensor(cls, data):
        return type(data) in _NESTED_NODE_TO_DTYPE

    @property
    def depth(self):
        """Number of ``Tensor`` levels, including the leaf tensor."""
        return self._root.depth

    def _to_py_tensor(self, data):
        node_type, _ = _NESTED_CPP_TYPES[self.dtype]
        root = node_type._from_outer_view(data, self.depth - 1)
        return NestedTensor._from_cpp(root)

    def _represent_element(self, element):
        if element.is_leaf:
            return _leaf_wrappers()[self.dtype](element.leaf)
        return NestedTensor._from_cpp(element)

    def _decay_value(self, val):
        node, _ = _nested_node(val, dtype=self.dtype)
        if node.depth != self.depth - 1:
            raise ValueError(
                f"NestedTensor element depth must be {self.depth - 1}, got {node.depth}"
            )
        return node

    def _is_tensor_rhs(self, val):
        return isinstance(val, NestedTensor) and val.depth == self.depth

    def _validate_setitem_value(self, entries, val):
        scalar_target = (
            len(entries) == self.ndim
            and all(isinstance(entry, int) for entry in entries)
        )
        if isinstance(val, NestedTensor):
            if val.dtype is not self.dtype:
                raise TypeError(
                    f"NestedTensor leaf dtype must be {self.dtype}, got {val.dtype}"
                )
            allowed_depths = {self.depth - 1} if scalar_target else {
                self.depth - 1, self.depth
            }
            if val.depth not in allowed_depths:
                expected = " or ".join(str(depth) for depth in sorted(allowed_depths))
                raise ValueError(
                    f"NestedTensor assignment depth must be {expected}, got {val.depth}"
                )
            return
        self._decay_value(val)

    def _get_valid_setitem_dtypes(self):
        return [NestedTensor, *set(_leaf_wrappers().values()), np.ndarray]

    def copy(self):
        return NestedTensor._from_cpp(self._root.copy())

    def __deepcopy__(self, memodict=None):
        return self.copy()

    def __repr__(self):
        return "Tensor<" * self.depth + self.dtype.name + ">" * self.depth
