"""Bug-scan regression tests for stablebear.io.load (issue #38).

Loading a truncated/corrupt single-tensor file used to surface a misleading
"Expected format type SingleObject ... but got format type SingleTensor" error,
because `load` blindly retried the single-object path after *any* tensor-load
error. It should instead re-raise the real corruption/EOF error and only fall
back to the object path for a genuine format-type mismatch.
"""

import io

import numpy as np
import pytest

import stablebear as sb
import stablebear.persistence as pers


_CORRUPTION_KEYWORDS = ("corrupt", "bytes", "eof", "truncat")


def test_truncated_tensor_surfaces_real_error_not_misleading_one():
    """A truncated tensor must report corruption, not the SingleObject mismatch."""
    X = sb.random.noisy_sin((3,), dtype=sb.pcf64)
    buf = io.BytesIO()
    sb.save(X, buf)
    data = buf.getvalue()

    # Chop off the tail so the header + format-type bytes survive but the
    # tensor payload is incomplete -- this is exactly what triggered the
    # misleading fallback message before the fix.
    truncated = io.BytesIO(data[: len(data) - 8])
    with pytest.raises(RuntimeError) as excinfo:
        sb.load(truncated)

    msg = str(excinfo.value)
    # The misleading message must be gone ...
    assert "format type SingleObject" not in msg
    # ... and the real cause must be reported.
    assert any(kw in msg.lower() for kw in _CORRUPTION_KEYWORDS)


def test_truncated_to_half_surfaces_real_error():
    """Same as above but truncated to half (matches the robustness suite case)."""
    X = sb.random.noisy_sin((3,), dtype=sb.pcf64)
    buf = io.BytesIO()
    sb.save(X, buf)
    data = buf.getvalue()

    truncated = io.BytesIO(data[: len(data) // 2])
    with pytest.raises(RuntimeError) as excinfo:
        sb.load(truncated)
    assert "format type SingleObject" not in str(excinfo.value)


def test_valid_pcf_object_still_round_trips():
    """A genuine single object must still load via the object fallback."""
    f = sb.Pcf(np.array([[0.0, 1.0], [1.0, 2.0], [3.0, 0.0]]))
    buf = io.BytesIO()
    sb.save(f, buf)
    buf.seek(0)
    g = sb.load(buf)
    assert isinstance(g, sb.Pcf)
    assert f == g


def test_valid_distance_matrix_object_still_round_trips():
    """Another single-object type round-trips through load."""
    dm = sb.DistanceMatrix(3, dtype=sb.float64)
    dm[1, 0] = 1.0
    dm[2, 0] = 2.0
    dm[2, 1] = 3.0
    buf = io.BytesIO()
    sb.save(dm, buf)
    buf.seek(0)
    dm2 = sb.load(buf)
    assert isinstance(dm2, sb.DistanceMatrix)
    np.testing.assert_allclose(dm.to_dense(), dm2.to_dense())


def test_valid_barcode_object_still_round_trips():
    bc = pers.Barcode(np.array([[0.0, 1.0], [0.5, 3.0]]))
    buf = io.BytesIO()
    sb.save(bc, buf)
    buf.seek(0)
    bc2 = sb.load(buf)
    assert isinstance(bc2, pers.Barcode)
    assert bc.is_isomorphic_to(bc2)


def test_valid_tensor_still_round_trips():
    """A valid tensor must still load correctly via the primary path."""
    X = sb.random.noisy_sin((4,), dtype=sb.pcf64)
    buf = io.BytesIO()
    sb.save(X, buf)
    buf.seek(0)
    Y = sb.load(buf)
    assert Y.shape == (4,)
    assert Y.dtype == sb.pcf64


def test_empty_bytes_still_raises():
    with pytest.raises((RuntimeError, ValueError, EOFError)):
        sb.load(io.BytesIO(b""))


def test_random_garbage_still_raises():
    with pytest.raises((RuntimeError, ValueError, EOFError)):
        sb.load(io.BytesIO(b"\x00\x01\x02\x03\xff\xfe\xfd" * 10))
