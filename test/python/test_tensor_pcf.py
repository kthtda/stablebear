import numpy as np
import numpy.testing as npt

import stablebear as sb


def test_pcf_tensor_basic():
    X = sb.zeros((2, 2))

    assert X.dtype == sb.pcf32

    data = np.array([[0, 3], [1, 5], [2, 0]])

    X[0, 0] = sb.Pcf(data, dtype=sb.pcf32)

    X00np = np.asarray(X[0, 0])

    npt.assert_equal(data, X00np)
