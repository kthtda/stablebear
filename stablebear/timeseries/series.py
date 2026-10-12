from __future__ import annotations

import numpy as np

from .. import _sb_cpp as cpp
from ..base_tensor import FloatTensor


class Times:
    """Finite, strictly increasing timestamps in one physical time unit.

    Parameters
    ----------
    values : array_like
        One-dimensional timestamps, converted to float64.
    unit : str, optional
        One of ``"as"``, ``"fs"``, ``"ps"``, ``"ns"``, ``"us"``, ``"ms"``,
        ``"s"``, ``"min"``, ``"h"``, and ``"d"``, or an alias such as
        ``"sec"``, ``"seconds"``, ``"hr"``, ``"\u00b5s"``, or ``"msec"``.
        Defaults to seconds. :attr:`unit` reports the short name.

    Raises
    ------
    ValueError
        If the timestamps are not one-dimensional, finite, and strictly
        increasing, or if the unit is unknown.
    """

    def __init__(self, values, unit: str = "s"):
        self._data = cpp.Times(np.asarray(values, dtype=np.float64), unit)

    @classmethod
    def _from_cpp(cls, data):
        times = cls.__new__(cls)
        times._data = data
        return times

    @property
    def values(self) -> FloatTensor:
        """A float64 copy of the timestamps; changing it leaves these unchanged."""
        return FloatTensor(self._data.values)

    @property
    def unit(self) -> str:
        """The time unit, such as ``"s"`` or ``"ms"``."""
        return self._data.unit

    def to(self, unit: str) -> Times:
        """Return the same timestamps in another unit.

        The conversion multiplies or divides each timestamp by an exact
        integer factor, so each result is rounded once. Only the factor
        between days and attoseconds exceeds 2^53 and is itself rounded.
        """
        return Times._from_cpp(self._data.to(unit))

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other):
        if not isinstance(other, Times):
            return NotImplemented
        return self._data == other._data

    def __repr__(self):
        return f"Times({np.asarray(self.values).tolist()!r}, unit={self.unit!r})"
