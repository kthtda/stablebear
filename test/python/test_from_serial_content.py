import numpy as np
import pytest

import stablebear as sb


def test_fsc_shapes():
    content = np.zeros((0, 2))
    enumeration = np.zeros((0, 2))

    X = sb.from_serial_content(content, enumeration)

    assert len(X.shape) == 1
    assert X.shape[0] == 0

    content = np.zeros((1000, 2))
    enumeration = np.zeros((10, 20, 30, 2), dtype=np.longlong)
    enumeration[:, :, :, 1] = 1  # To make every stop > start

    X = sb.from_serial_content(content, enumeration)

    assert len(X.shape) == 3
    assert X.shape[0] == 10
    assert X.shape[1] == 20
    assert X.shape[2] == 30


def test_fsc_single():
    content = np.array([[0.0, 10.0], [10.0, 20.0], [20.0, 30.0]])
    enumeration = np.array([[0, 3]])

    X = sb.from_serial_content(content, enumeration)

    assert X[0] == sb.Pcf(content[0:3])


def test_fsc_multiple():
    content = np.array(
        [[0.0, 10.0], [10.0, 20.0], [20.0, 30.0], [0.0, 50.0], [10.0, 60.0]]
    )
    enumeration = np.array([[0, 3], [3, 5]])

    X = sb.from_serial_content(content, enumeration)

    assert X[0] == sb.Pcf(content[0:3])
    assert X[1] == sb.Pcf(content[3:5])


def test_fsc_multidim():
    content = np.array(
        [
            [0.0, 10.0],
            [10.0, 20.0],
            [20.0, 30.0],
            [0.0, 50.0],
            [10.0, 60.0],
            [0.0, 5.0],
            [0.0, 6.0],
            [0.0, 7.0],
            [0.0, 8.0],
            [0.0, 0.0],
            [0.0, 2.0],
            [1.0, 3.0],
            [3.0, 0.0],
            [0.0, 10.0],
            [0.0, 100.0],
        ]
    )

    enumeration = np.array([[[0, 3], [3, 5]], [[5, 9], [9, 10]], [[10, 13], [13, 15]]])

    X = sb.from_serial_content(content, enumeration)

    assert X.shape == (3, 2)

    # from_serial_content is a loose deserialization constructor and may produce
    # PCFs with duplicate breakpoint times (e.g. the [5:9] / [13:15] slices here
    # are all t=0), which the strict public Pcf(array) constructor rejects. Test
    # the slicing by comparing the produced PCF's breakpoints to the source
    # slice directly, rather than rebuilding the expected value via Pcf(...).
    for i in range(3):
        for j in range(2):
            start, stop = enumeration[i, j, 0], enumeration[i, j, 1]
            np.testing.assert_array_equal(np.asarray(X[i, j]), content[start:stop])


def test_fsc_stop_le_start_throws():
    content = np.array(
        [[0.0, 10.0], [10.0, 20.0], [20.0, 30.0], [0.0, 50.0], [10.0, 60.0]]
    )
    enumeration = np.array([[3, 0], [3, 5]])

    with pytest.raises(ValueError):
        X = sb.from_serial_content(content, enumeration)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_fsc_stop_overflow_raises(dtype):
    """stop beyond content length must raise cleanly, not read OOB heap memory.

    Regression for #37: stop > content.shape(0) was unchecked, so the loop read
    past the content buffer and returned uninitialized heap garbage (small
    overflow) or segfaulted (large overflow), exit 0 / no exception.
    """
    content = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=dtype)
    enumeration = np.array([[0, 20]], dtype=np.int64)
    with pytest.raises((ValueError, IndexError)):
        sb.from_serial_content(content, enumeration)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_fsc_negative_start_raises(dtype):
    """Negative start bypassed the start>=stop guard and read negative offsets.

    Regression for #37 case (C): enum [[-1000, 1]] satisfied start < stop yet
    read OOB; it must raise.
    """
    content = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=dtype)
    enumeration = np.array([[-1000, 1]], dtype=np.int64)
    with pytest.raises((ValueError, IndexError)):
        sb.from_serial_content(content, enumeration)


def test_fsc_stop_equals_length_ok():
    """stop == content.shape(0) is in-bounds and must still succeed."""
    content = np.array([[0.0, 10.0], [10.0, 20.0], [20.0, 30.0]])
    enumeration = np.array([[0, 3]])
    X = sb.from_serial_content(content, enumeration)
    assert X[0] == sb.Pcf(content[0:3])


def test_fsc_dtypes():
    content32 = np.array(
        [[0.0, 10.0], [10.0, 20.0], [20.0, 30.0], [0.0, 50.0], [10.0, 60.0]],
        dtype=np.float32,
    )
    content64 = np.array(
        [[0.0, 10.0], [10.0, 20.0], [20.0, 30.0], [0.0, 50.0], [10.0, 60.0]],
        dtype=np.float64,
    )
    enumeration = np.array([[0, 3], [3, 5]])

    X32 = sb.from_serial_content(content32, enumeration)
    assert X32.dtype == sb.pcf32

    X64 = sb.from_serial_content(content64, enumeration)
    assert X64.dtype == sb.pcf64

    X32_64 = sb.from_serial_content(content32, enumeration, dtype=sb.pcf64)
    assert X32_64.dtype == sb.pcf64

    X64_32 = sb.from_serial_content(content32, enumeration, dtype=sb.pcf32)
    assert X64_32.dtype == sb.pcf32
