"""
import pytest
import stablebear as sb

def test_zeros():
    Z = sb.zeros((1, 2, 3))

    assert len(Z.shape) == 3
    assert Z.shape[0] == 1
    assert Z.shape[1] == 2
    assert Z.shape[2] == 3

@pytest.mark.parametrize("gpu", [True, False])
def test_pdist_zeros(gpu):
    Z = sb.zeros((2,))
    D = sb.pdist(Z)

    assert D.shape == (2, 2)
    assert D[0][0] == 0.
    assert D[0][1] == 0.
    assert D[1][0] == 0.
    assert D[1][1] == 0.
"""
