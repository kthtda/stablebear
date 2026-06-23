import numpy as np

import stablebear as sb
import stablebear._sb_cpp as cpp


def test_numpy_tensor_create_gives_correct_cpp_type():
    Xnp = np.zeros((10, 20), dtype=np.float32)
    X = sb.FloatTensor(Xnp)
    assert isinstance(X._data, cpp.Float32Tensor)

    Xnp = np.zeros((10, 20), dtype=np.float64)
    X = sb.FloatTensor(Xnp)
    assert isinstance(X._data, cpp.Float64Tensor)
