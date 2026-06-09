"""Tests for pdist and l2_kernel on the CUDA path.

Skipped when no GPU is available, unless SB_REQUIRE_CUDA=1 is set,
in which case missing CUDA causes a hard failure.
"""

import os

import numpy as np
import pytest

import stablebear as sb
from stablebear import _sb_cpp as cpp
from stablebear.distance_matrix import DistanceMatrix
from stablebear.symmetric_matrix import SymmetricMatrix

_has_cuda = cpp._build_type() == "CUDA" and cpp.get_ngpus() > 0
_require_cuda = os.environ.get("SB_REQUIRE_CUDA", "0") == "1"

if _require_cuda and not _has_cuda:
    pytest.fail(
        "SB_REQUIRE_CUDA=1 but CUDA is not available "
        f"(build_type={cpp._build_type()}, ngpus={cpp.get_ngpus()})",
        pytrace=False,
    )

requires_cuda = pytest.mark.skipif(
    not _has_cuda,
    reason="Requires CUDA build with at least one GPU",
)


@pytest.fixture(autouse=True)
def _force_cuda():
    """Force the CUDA path and enable verbose device reporting."""
    cpp.set_cuda_threshold(0)
    cpp.set_device_verbose(True)
    yield
    cpp.set_device_verbose(False)
    cpp.set_cuda_threshold(500)


def _assert_ran_on_cuda(captured_err_out):
    """Assert that the captured output indicates CUDA execution."""
    assert "CUDA" in captured_err_out, (
        f"Expected CUDA execution but got: {captured_err_out!r}"
    )


@requires_cuda
def test_pdist_cuda_returns_distance_matrix(capfd):
    X = sb.zeros((3,))
    D = sb.pdist(X)
    _assert_ran_on_cuda(capfd.readouterr().out)
    assert isinstance(D, DistanceMatrix)
    assert D.size == 3


@requires_cuda
def test_pdist_cuda_correct_values(capfd):
    X = sb.zeros((2,), dtype=sb.pcf64)

    X[0] = sb.Pcf(np.array([[0.0, 10.0], [2.0, 5.0], [3.0, 0.0]]))
    X[1] = sb.Pcf(np.array([[0.0, 5.0], [6.0, 0.0]]))

    D = sb.pdist(X)
    _assert_ran_on_cuda(capfd.readouterr().out)

    assert D.size == 2
    assert D[0, 0] == 0.0
    assert D[0, 1] == pytest.approx(2 * 5 + 3 * 5)
    assert D[1, 0] == D[0, 1]
    assert D[1, 1] == 0.0


@requires_cuda
def test_pdist_cuda_to_dense(capfd):
    X = sb.zeros((2,), dtype=sb.pcf64)

    X[0] = sb.Pcf(np.array([[0.0, 10.0], [2.0, 5.0], [3.0, 0.0]]))
    X[1] = sb.Pcf(np.array([[0.0, 5.0], [6.0, 0.0]]))

    D = sb.pdist(X)
    _assert_ran_on_cuda(capfd.readouterr().out)
    dense = D.to_dense()

    assert isinstance(dense, np.ndarray)
    assert dense.shape == (2, 2)
    assert dense[0, 1] == pytest.approx(2 * 5 + 3 * 5)


@requires_cuda
def test_l2_kernel_cuda_returns_symmetric_matrix(capfd):
    X = sb.zeros((3,))
    K = sb.l2_kernel(X)
    _assert_ran_on_cuda(capfd.readouterr().out)
    assert isinstance(K, SymmetricMatrix)
    assert K.size == 3


@requires_cuda
def test_l2_kernel_cuda_correct_values(capfd):
    X = sb.zeros((2,), dtype=sb.pcf64)

    X[0] = sb.Pcf(np.array([[0.0, 4.0], [3.0, 0.0]]))
    X[1] = sb.Pcf(np.array([[0.0, 2.0], [3.0, 0.0]]))

    K = sb.l2_kernel(X)
    _assert_ran_on_cuda(capfd.readouterr().out)

    assert K.size == 2
    assert K[0, 0] == pytest.approx(48.0)
    assert K[0, 1] == pytest.approx(24.0)
    assert K[1, 0] == K[0, 1]
    assert K[1, 1] == pytest.approx(12.0)


@requires_cuda
def test_l2_kernel_cuda_to_dense(capfd):
    X = sb.zeros((2,), dtype=sb.pcf64)

    X[0] = sb.Pcf(np.array([[0.0, 4.0], [3.0, 0.0]]))
    X[1] = sb.Pcf(np.array([[0.0, 2.0], [3.0, 0.0]]))

    K = sb.l2_kernel(X)
    _assert_ran_on_cuda(capfd.readouterr().out)
    dense = K.to_dense()

    assert isinstance(dense, np.ndarray)
    assert dense.shape == (2, 2)
    assert dense[0, 0] == pytest.approx(48.0)
    assert dense[0, 1] == pytest.approx(24.0)


@requires_cuda
def test_pdist_lp_cuda_correct_values(capfd):
    X = sb.zeros((2,), dtype=sb.pcf64)

    X[0] = sb.Pcf(np.array([[0.0, 4.0], [1.0, 0.0]]))
    X[1] = sb.Pcf(np.array([[0.0, 1.0], [1.0, 0.0]]))

    D = sb.pdist(X, p=3)
    _assert_ran_on_cuda(capfd.readouterr().out)

    assert isinstance(D, DistanceMatrix)
    # ||f - g||_3 = (|4-1|^3 * 1)^(1/3) = 3
    assert D[0, 1] == pytest.approx(3.0)
