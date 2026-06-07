"""Red-until-fixed regression tests for KNOWN, unfixed IO/serialization bugs.

These tests were authored by the bug scan (see ``bug-scan-findings.md`` at the
repo root). Each asserts the CORRECT/intended behavior, so it FAILS today and
will pass once the underlying defect is fixed. Do NOT weaken a test to make it
green -- fix the bug instead.

Covered defects (area key: io):
  * #36 ``from_serial_content`` reads out of bounds on ``content`` when an
    enumeration entry's ``start``/``stop`` are not bounded against
    ``content.shape(0)`` -- silent heap garbage for small overflow, hard
    SIGSEGV for large overflow, and a negative ``start`` that bypasses the
    existing ``start >= stop`` guard. Tested via subprocess helpers because one
    case hard-crashes the interpreter.
  * #37 ``mpcf.load`` surfaces a misleading "Expected format type SingleObject
    ... but got SingleTensor" error for a truncated/corrupt *tensor* file,
    masking the real "file may be corrupted" parse error.
"""

import io

import numpy as np
import pytest

import masspcf as mpcf

from _bugscan_support import assert_clean_raises


# --- Bug #36: from_serial_content out-of-bounds read -------------------------


def test_from_serial_content_positive_overflow_raises():
    """stop beyond content length must raise cleanly, not return heap garbage.

    Case (A) from the scan: small overflow currently returns a Pcf whose extra
    rows are uninitialized heap memory (exit 0, no error).
    """
    # BUG: #36 from_serial_content reads OOB when stop > content.shape(0)
    # Expected: a clean ValueError/IndexError (matching the existing
    #           start >= stop guard); element i is content[start:stop] and the
    #           docstring makes no allowance for OOB reads.
    # Observed today: returns a (20, 2) Pcf, rows 2..19 are uninitialized heap
    #                 garbage, exit 0 (no exception).
    for dtype in ("np.float64", "np.float32"):
        assert_clean_raises(
            f"""
            content = np.array([[0.0, 1.0], [2.0, 3.0]], dtype={dtype})
            enumeration = np.array([[0, 20]], dtype=np.int64)
            mpcf.from_serial_content(content, enumeration)
            """,
            ("ValueError", "IndexError"),
        )


def test_from_serial_content_large_overflow_raises_not_segfault():
    """A huge stop must raise cleanly instead of segfaulting (case B).

    This case HARD-CRASHES (SIGSEGV, exit 139) in-process, so it is exercised
    only through the subprocess helper.
    """
    # BUG: #36 from_serial_content walks off the mapped heap for large overflow
    # Expected: a clean ValueError/IndexError on stop > content.shape(0).
    # Observed today: Segmentation fault (shell exit 139, core dumped), no
    #                 Python traceback -- an unchecked OOB read, not an
    #                 allocation failure.
    assert_clean_raises(
        """
        content = np.zeros((1, 2), dtype=np.float64)
        enumeration = np.array([[0, 10**7]], dtype=np.int64)
        mpcf.from_serial_content(content, enumeration)
        """,
        ("ValueError", "IndexError"),
    )


def test_from_serial_content_negative_start_raises():
    """A negative start must raise; it currently bypasses the start>=stop guard.

    Case (C): -1000 < 1 so the existing ``start >= stop`` check does not fire,
    and the loop reads negative offsets into content.
    """
    # BUG: #36 negative start bypasses the only existing bounds guard
    # Expected: a clean ValueError/IndexError (start must satisfy 0 <= start).
    # Observed today: returns a (1001, 2) Pcf of pure garbage, exit 0.
    assert_clean_raises(
        """
        content = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
        enumeration = np.array([[-1000, 1]], dtype=np.int64)
        mpcf.from_serial_content(content, enumeration)
        """,
        ("ValueError", "IndexError"),
    )


# --- Bug #37: misleading load() error for truncated tensor files -------------


def test_load_truncated_tensor_error_reflects_corruption():
    """A truncated tensor file should surface the real corruption cause.

    Does not crash (clean RuntimeError, exit 0), so this is a normal in-process
    test.
    """
    # BUG: #37 load() reports the wrong format type for a corrupt tensor file
    # Expected: the surfaced error reflects the real cause (the file is a
    #           corrupt/truncated SingleTensor -> "file may be corrupted"/EOF),
    #           NOT a claim that the format type is SingleObject.
    # Observed today: RuntimeError "Expected format type SingleObject for this
    #                 operation but got format type SingleTensor" -- the exact
    #                 opposite of the truth, masking the genuine parse error.
    X = mpcf.random.noisy_sin((3,), dtype=mpcf.pcf64)
    buf = io.BytesIO()
    mpcf.save(X, buf)
    good = buf.getvalue()
    trunc = good[:200]  # truncated SingleTensor file (full is ~1081 bytes)

    with pytest.raises(Exception) as excinfo:
        mpcf.load(io.BytesIO(trunc))

    msg = str(excinfo.value)
    # The diagnostic must not misdirect the user about the file's format type.
    assert "Expected format type SingleObject" not in msg, (
        "load() surfaced the misleading SingleObject mismatch error for a "
        f"corrupt SingleTensor file; message was: {msg!r}"
    )
    # And it should point at the real cause (truncation/corruption/EOF).
    assert any(
        token in msg.lower() for token in ("corrupt", "byte", "eof", "truncat")
    ), f"load() error should reflect corruption; message was: {msg!r}"
