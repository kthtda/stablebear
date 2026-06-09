"""Shared fixtures for stablebear Python tests."""

import os

import pytest

import stablebear as sb
from stablebear import _sb_cpp as cpp

_has_cuda = cpp._build_type() == "CUDA" and cpp.get_ngpus() > 0
_require_cuda = os.environ.get("SB_REQUIRE_CUDA", "0") == "1"

if _require_cuda and not _has_cuda:
    pytest.fail(
        "SB_REQUIRE_CUDA=1 but CUDA is not available "
        f"(build_type={cpp._build_type()}, ngpus={cpp.get_ngpus()})",
        pytrace=False,
    )


def _cuda_param():
    return pytest.param("cuda", marks=pytest.mark.skipif(
        not _has_cuda, reason="Requires CUDA build with at least one GPU"))


@pytest.fixture(params=["cpu", _cuda_param()])
def device(request):
    """Parametrize a test to run on both CPU and CUDA.

    On CPU: forces CPU execution.
    On CUDA: lowers the CUDA threshold so the GPU path is taken.
    Skips the CUDA variant when no GPU is available.
    """
    if request.param == "cuda":
        cpp.force_cpu(False)
        cpp.set_cuda_threshold(0)
    else:
        cpp.force_cpu(True)

    yield request.param

    cpp.force_cpu(False)
    cpp.set_cuda_threshold(500)


@pytest.fixture(params=[sb.pcf32, sb.pcf64], ids=["pcf32", "pcf64"])
def pcf_dtype(request):
    """Parametrize a test to run with both pcf32 and pcf64."""
    return request.param
