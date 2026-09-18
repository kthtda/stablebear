from __future__ import annotations

import numpy as np

from . import _sb_cpp as cpp
from ._tensor_base import Tensor, _infer_shape_and_flatten
from .typing import (
    _NP_TO_SB,
    float32,
    float64,
    int32,
    int64,
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
}
_NESTED_CPP_TO_DTYPE = {
    tensor_type: dtype
    for dtype, (_, tensor_type) in _NESTED_CPP_TYPES.items()
}


def _leaf_tensor(data, dtype=None):
    from .base_tensor import FloatTensor, IntTensor

    if isinstance(data, (FloatTensor, IntTensor)):
        if dtype is not None and data.dtype is not dtype:
            raise TypeError(
                f"NestedTensor leaf dtype must be {dtype}, got {data.dtype}"
            )
        leaf = data
    elif isinstance(data, np.ndarray):
        if dtype is None:
            dtype = _NP_TO_SB.get(data.dtype.type)
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
        else:
            raise TypeError(f"Unsupported NestedTensor leaf dtype {dtype}")
    else:
        raise TypeError(
            "NestedTensor leaves must be numeric Tensor or NumPy array objects"
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
        node_type, _ = _NESTED_CPP_TYPES[value.dtype]
        return node_type(value._data, value.depth - 1), value.dtype

    leaf = _leaf_tensor(value, dtype=dtype)
    node_type, _ = _NESTED_CPP_TYPES[leaf.dtype]
    return node_type(leaf._data), leaf.dtype


def _reconstruct_nested_tensor(shape, values, depth, dtype):
    _, tensor_type = _NESTED_CPP_TYPES[dtype]
    tensor = tensor_type(cpp.Shape(list(shape)))
    for index, value in zip(np.ndindex(*shape), values):
        tensor._set_element(list(index), value)
    return NestedTensor(tensor, depth=depth)


class NestedTensor(Tensor):
    """A recursively nested tensor whose leaf tensors share one numeric dtype."""

    def __init__(self, data, *, dtype=None, depth=None):
        from .base_tensor import FloatTensor, IntTensor

        super().__init__()
        if dtype is not None and dtype not in _NESTED_CPP_TYPES:
            raise TypeError(f"Unsupported NestedTensor leaf dtype {dtype}")

        if isinstance(data, NestedTensor):
            if dtype is not None and data.dtype is not dtype:
                raise TypeError(
                    f"NestedTensor leaf dtype must be {dtype}, got {data.dtype}"
                )
            source_depth = data.depth
            dtype = data.dtype
            data = data._data
            depth = source_depth if depth is None else depth
        elif isinstance(data, (FloatTensor, IntTensor, np.ndarray)):
            leaf = _leaf_tensor(data, dtype=dtype)
            dtype = leaf.dtype
            node_type, tensor_type = _NESTED_CPP_TYPES[dtype]
            tensor = tensor_type(cpp.Shape([]))
            tensor._set_element([], node_type(leaf._data))
            data = tensor
            depth = 2
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
                _, tensor_type = _NESTED_CPP_TYPES[dtype]
                data = tensor_type(cpp.Shape(list(shape or (0,))))
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
                depth = inferred_depth
                _, tensor_type = _NESTED_CPP_TYPES[dtype]
                tensor = tensor_type(cpp.Shape([len(nodes)]))
                for i, node in enumerate(nodes):
                    tensor._set_element([i], node)
                if shape != (len(nodes),):
                    tensor = tensor.reshape(list(shape))
                data = tensor
        elif type(data) in _NESTED_CPP_TO_DTYPE:
            actual_dtype = _NESTED_CPP_TO_DTYPE[type(data)]
            if dtype is not None and actual_dtype is not dtype:
                raise TypeError(
                    f"NestedTensor leaf dtype must be {dtype}, got {actual_dtype}"
                )
            dtype = actual_dtype
        else:
            raise TypeError(f"Cannot create NestedTensor from {type(data)}")

        if depth is None:
            shape = tuple(data.shape)
            if any(size == 0 for size in shape):
                raise ValueError("Empty NestedTensor requires an explicit depth")
            depth = data._get_element([0] * len(shape)).depth + 1

        self._data = data
        self.dtype = dtype
        self._depth = depth

    @classmethod
    def _from_cpp(cls, data):
        shape = tuple(data.shape)
        depth = 2 if any(size == 0 for size in shape) else None
        return cls(data, depth=depth)

    @classmethod
    def _is_cpp_nested_tensor(cls, data):
        return type(data) in _NESTED_CPP_TO_DTYPE

    @property
    def depth(self):
        """Number of ``Tensor`` levels, including the leaf tensor."""
        return self._depth

    def _to_py_tensor(self, data):
        return NestedTensor(data, depth=self.depth)

    def _represent_element(self, element):
        from .base_tensor import FloatTensor, IntTensor

        if element.is_leaf:
            wrapper = (
                FloatTensor
                if self.dtype in (float32, float64)
                else IntTensor
            )
            return wrapper(element.leaf)
        return NestedTensor(element.nested, depth=element.depth)

    def _decay_value(self, val):
        node, _ = _nested_node(val, dtype=self.dtype)
        if node.depth != self.depth - 1:
            raise ValueError(
                f"NestedTensor element depth must be {self.depth - 1}, got {node.depth}"
            )
        return node

    def _get_valid_setitem_dtypes(self):
        from .base_tensor import FloatTensor, IntTensor

        return [NestedTensor, FloatTensor, IntTensor, np.ndarray]

    def copy(self):
        shape = tuple(self.shape)
        values = [
            self._data._get_element(list(index)).copy()
            for index in np.ndindex(*shape)
        ]
        return _reconstruct_nested_tensor(
            shape, values, self.depth, self.dtype
        )

    def __deepcopy__(self, memodict=None):
        return self.copy()

    def __repr__(self):
        return "Tensor<" * self.depth + self.dtype.name + ">" * self.depth
