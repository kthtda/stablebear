#    Copyright 2024-2026 Bjorn Wehlin
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from __future__ import annotations

from .tensor import FloatTensor
from .distance_matrix import DistanceMatrix
from .symmetric_matrix import SymmetricMatrix


def allclose(a, b, atol=1e-8, rtol=1e-5) -> bool:
    r"""Test whether two objects are element-wise equal within a tolerance.

    Returns ``True`` when, for every pair of corresponding elements
    :math:`a_i` and :math:`b_i`,

    .. math::

        |a_i - b_i| \leq \texttt{atol} + \texttt{rtol} \cdot |b_i|.

    Parameters
    ----------
    a : FloatTensor | DistanceMatrix | SymmetricMatrix
        First object.
    b : FloatTensor | DistanceMatrix | SymmetricMatrix
        Second object (must be the same type as *a*).
    atol : float, optional
        Absolute tolerance, by default 1e-8.
    rtol : float, optional
        Relative tolerance, by default 1e-5.

    Returns
    -------
    bool

    Raises
    ------
    TypeError
        If the inputs are not a supported type or are not the same type.
    """
    if isinstance(a, FloatTensor) and isinstance(b, FloatTensor):
        return a._data.allclose(b._data, atol=atol, rtol=rtol)
    elif isinstance(a, DistanceMatrix) and isinstance(b, DistanceMatrix):
        return a._data.allclose(b._data, atol=atol, rtol=rtol)
    elif isinstance(a, SymmetricMatrix) and isinstance(b, SymmetricMatrix):
        return a._data.allclose(b._data, atol=atol, rtol=rtol)
    else:
        raise TypeError(
            f"allclose requires two objects of the same supported type "
            f"(FloatTensor, DistanceMatrix, or SymmetricMatrix), "
            f"got {type(a).__name__} and {type(b).__name__}"
        )
