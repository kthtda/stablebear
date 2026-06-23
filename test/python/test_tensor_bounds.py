import pytest

import stablebear as sb


def test_tensor_get_oob_item_raises():
    X = sb.zeros((10, 10), dtype=sb.float64)

    with pytest.raises(IndexError):
        f = X[10, 5]


def test_tensor_set_oob_item_raises():
    X = sb.zeros((10, 10), dtype=sb.float64)

    with pytest.raises(IndexError):
        X[10, 5] = 10.0
