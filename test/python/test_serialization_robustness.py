"""Robustness tests for serialization: corrupted data, truncation, edge cases."""

import io

import numpy as np
import pytest

import stablebear as sb
import stablebear.persistence as pers


# --- Corrupted data ---


def test_load_from_empty_bytes_raises():
    """Loading from empty bytes should raise, not crash."""
    buf = io.BytesIO(b"")
    with pytest.raises((RuntimeError, ValueError, EOFError)):
        sb.load(buf)


def test_load_from_random_bytes_raises():
    """Loading from random garbage should raise, not crash."""
    buf = io.BytesIO(b"\x00\x01\x02\x03\xff\xfe\xfd" * 10)
    with pytest.raises((RuntimeError, ValueError, EOFError)):
        sb.load(buf)


def test_load_truncated_data_raises():
    """Saving then truncating should raise on load."""
    X = sb.random.noisy_sin((3,), dtype=sb.pcf64)
    buf = io.BytesIO()
    sb.save(X, buf)
    data = buf.getvalue()

    # Truncate to half
    truncated = io.BytesIO(data[: len(data) // 2])
    with pytest.raises((RuntimeError, ValueError, EOFError)):
        sb.load(truncated)


# --- Edge-case tensors ---


def test_save_load_single_element_tensor():
    """Roundtrip a (1,) tensor."""
    X = sb.zeros((1,), dtype=sb.pcf64)
    X[0] = sb.Pcf(np.array([[0.0, 42.0], [1.0, 0.0]]))

    buf = io.BytesIO()
    sb.save(X, buf)
    buf.seek(0)
    Y = sb.load(buf)

    assert Y.shape == (1,)
    assert Y[0] == X[0]


def test_save_load_high_dimensional_tensor():
    """Roundtrip a 4D tensor."""
    X = sb.zeros((2, 2, 2, 2), dtype=sb.pcf32)

    buf = io.BytesIO()
    sb.save(X, buf)
    buf.seek(0)
    Y = sb.load(buf)

    assert Y.shape == (2, 2, 2, 2)
    assert Y.dtype == sb.pcf32


def test_save_load_pcf_object():
    """Roundtrip a single Pcf object."""
    f = sb.Pcf(np.array([[0.0, 1.0], [1.0, 2.0], [3.0, 0.0]]))

    buf = io.BytesIO()
    sb.save(f, buf)
    buf.seek(0)
    g = sb.load(buf)

    assert isinstance(g, sb.Pcf)
    assert f == g


def test_save_load_barcode_object():
    """Roundtrip a single Barcode object."""
    bc = pers.Barcode(np.array([[0.0, 1.0], [0.5, 3.0]]))

    buf = io.BytesIO()
    sb.save(bc, buf)
    buf.seek(0)
    bc2 = sb.load(buf)

    assert isinstance(bc2, pers.Barcode)
    assert bc.is_isomorphic_to(bc2)


def test_save_load_distance_matrix():
    """Roundtrip a DistanceMatrix object."""
    dm = sb.DistanceMatrix(4, dtype=sb.float64)
    dm[1, 0] = 1.0
    dm[2, 0] = 2.0
    dm[2, 1] = 3.0
    dm[3, 0] = 4.0
    dm[3, 1] = 5.0
    dm[3, 2] = 6.0

    buf = io.BytesIO()
    sb.save(dm, buf)
    buf.seek(0)
    dm2 = sb.load(buf)

    assert isinstance(dm2, sb.DistanceMatrix)
    assert dm2.size == 4
    np.testing.assert_allclose(dm.to_dense(), dm2.to_dense())


def test_save_load_empty_barcode_tensor():
    """Roundtrip a barcode tensor where all barcodes are empty."""
    bcs = sb.zeros((3,), dtype=sb.barcode64)
    buf = io.BytesIO()
    sb.save(bcs, buf)
    buf.seek(0)
    bcs2 = sb.load(buf)
    assert bcs2.shape == (3,)
    for i in range(3):
        assert len(bcs2[i]) == 0

