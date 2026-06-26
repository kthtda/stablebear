"""Validation of the runtime-config setters in ``stablebear.system`` (issue #101).

These setters used to forward their integer argument straight to the C++
backend with no validation. A negative argument triggered pybind's
"incompatible function arguments" overload dump (the C++ params are unsigned),
and a zero worker count made taskflow raise a ``RuntimeError`` referencing an
internal source path. They now raise a clean ``ValueError`` naming the
parameter and its bound.

Only the error paths are exercised here: validation happens *before* the C++
call, so a rejected call does not mutate any global runtime state. The setters
are deliberately never called with a valid value, since that would change the
global thread / GPU configuration for the rest of the test suite.
"""

import pytest

import stablebear as sb

# Things that must never leak into the user-facing message.
_LEAKY = ("incompatible function arguments", "taskflow", ".hpp", ".cpp", "/3rd/")


def _assert_clean(excinfo, *expected_substrings):
    msg = str(excinfo.value)
    lower = msg.lower()
    for leak in _LEAKY:
        assert leak.lower() not in lower, f"leaky detail {leak!r} in message: {msg!r}"
    for sub in expected_substrings:
        assert sub in msg, f"expected {sub!r} in message: {msg!r}"


# --- worker-count setters: require n >= 1 (0 and negatives rejected) ---


@pytest.mark.parametrize("n", [-1, 0])
def test_limit_cpus_rejects_non_positive(n):
    with pytest.raises(ValueError, match=r"limit_cpus") as excinfo:
        sb.system.limit_cpus(n)
    _assert_clean(excinfo, "limit_cpus", ">= 1")


@pytest.mark.parametrize("n", [-1, 0])
def test_limit_gpus_rejects_non_positive(n):
    with pytest.raises(ValueError, match=r"limit_gpus") as excinfo:
        sb.system.limit_gpus(n)
    _assert_clean(excinfo, "limit_gpus", ">= 1")


# --- threshold setters: require n >= 0 (negatives rejected) ---


def test_set_cuda_threshold_rejects_negative():
    with pytest.raises(ValueError, match=r"set_cuda_threshold") as excinfo:
        sb.system.set_cuda_threshold(-1)
    _assert_clean(excinfo, "set_cuda_threshold", ">= 0")


def test_set_parallel_eval_threshold_rejects_negative():
    with pytest.raises(ValueError, match=r"set_parallel_eval_threshold") as excinfo:
        sb.system.set_parallel_eval_threshold(-1)
    _assert_clean(excinfo, "set_parallel_eval_threshold", ">= 0")


def test_set_min_block_side_rejects_negative():
    with pytest.raises(ValueError, match=r"set_min_block_side") as excinfo:
        sb.system.set_min_block_side(-1)
    _assert_clean(excinfo, "set_min_block_side", ">= 0")


# --- block-dimension setter: require each dimension >= 1 ---


def test_set_block_size_rejects_non_positive_x():
    with pytest.raises(ValueError, match=r"set_block_size: x") as excinfo:
        sb.system.set_block_size(0, 16)
    _assert_clean(excinfo, "x", ">= 1")


def test_set_block_size_rejects_negative_x():
    with pytest.raises(ValueError, match=r"set_block_size: x") as excinfo:
        sb.system.set_block_size(-1, 16)
    _assert_clean(excinfo, "x", ">= 1")


def test_set_block_size_rejects_non_positive_y():
    with pytest.raises(ValueError, match=r"set_block_size: y") as excinfo:
        sb.system.set_block_size(16, 0)
    _assert_clean(excinfo, "y", ">= 1")


def test_set_block_size_rejects_negative_y():
    with pytest.raises(ValueError, match=r"set_block_size: y") as excinfo:
        sb.system.set_block_size(16, -1)
    _assert_clean(excinfo, "y", ">= 1")
