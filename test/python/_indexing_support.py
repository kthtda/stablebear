#  Copyright 2024-2026 Bjorn Wehlin
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Helpers for the tensor-indexing parity tests.

Some indexing operations currently dereference an invalid (negative or
out-of-buffer) data offset. Reading such a view returns garbage and writing
to it corrupts memory or raises SIGSEGV, which would abort the whole pytest
session. To keep the suite runnable while still *asserting on the result*
(not merely executing the expression), those cases run in an isolated
subprocess: the snippet performs the correctness check internally and the
parent test asserts the subprocess exited cleanly.

These tests assert the correct NumPy-parity behavior, so they fail (or the
subprocess crashes / exits non-zero) until the underlying bugs are fixed,
at which point they pass.
"""

import subprocess
import sys
import textwrap

# Reference arrays + a fresh-tensor factory available to every isolated snippet.
PRELUDE = textwrap.dedent(
    """
    import sys
    import numpy as np
    from masspcf.tensor import FloatTensor, IntTensor, BoolTensor

    a = np.arange(24., dtype=np.float64).reshape(4, 6)
    v = np.arange(6., dtype=np.float64)
    b = np.arange(24., dtype=np.float64).reshape(2, 3, 4)

    def F(x=None):
        '''Fresh FloatTensor over a copy of `a` (or `x`).'''
        return FloatTensor((a if x is None else x).copy())
    """
)


def run_isolated(body: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
    """Run ``PRELUDE + body`` in a fresh interpreter and return the result.

    The snippet is expected to perform its own assertions and exit 0 on
    success. A return code of 0 means the correct behavior held; a negative
    return code means the child was killed by a signal (e.g. -11 == SIGSEGV);
    a positive return code means a Python exception / assertion failure.
    """
    code = PRELUDE + "\n" + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def assert_isolated_ok(body: str) -> None:
    """Assert that an isolated snippet exercising correct behavior exits cleanly.

    Fails with the child's stdout/stderr (or the killing signal) attached, so a
    SIGSEGV or a failed in-child assertion produces a readable test failure
    rather than aborting the pytest session.
    """
    result = run_isolated(body)
    if result.returncode != 0:
        if result.returncode < 0:
            reason = f"child killed by signal {-result.returncode} (e.g. -11 == SIGSEGV)"
        else:
            reason = f"child exited with code {result.returncode}"
        raise AssertionError(
            f"{reason}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )


# --- In-process NumPy-oracle helpers --------------------------------------
# NumPy is the oracle: the masspcf result must match what NumPy produces for
# the same index expression on the same array. Used for the cases that raise
# or return a wrong value in-process (i.e. do NOT dereference an invalid
# offset); the memory-unsafe cases use the isolated-subprocess helpers above.

def ref_array():
    """A reusable (4, 6) float64 reference array (a fresh copy each call)."""
    import numpy as np
    return np.arange(24.0, dtype=np.float64).reshape(4, 6)


def assert_getitem_matches(arr, index):
    """``np.asarray(FloatTensor(arr)[index])`` must equal ``arr[index]``.

    Compares shape and values against NumPy. ``index`` may be anything that is
    a legal NumPy index (int, slice, tuple, Ellipsis, None, list, ndarray, ...).
    """
    import numpy as np
    from masspcf.tensor import FloatTensor

    a = np.asarray(arr, dtype=np.float64)
    expected = a[index]
    got = np.asarray(FloatTensor(a.copy())[index])
    assert got.shape == expected.shape, f"shape {got.shape} != numpy {expected.shape}"
    np.testing.assert_array_equal(got, expected)


def assert_setitem_matches(arr, index, value):
    """``FloatTensor(arr)[index] = value`` must mutate like ``arr[index] = value``.

    ``value`` may be a scalar or an ndarray; an ndarray is wrapped in a
    ``FloatTensor`` for the masspcf side (its RHS contract).
    """
    import numpy as np
    from masspcf.tensor import FloatTensor

    a = np.asarray(arr, dtype=np.float64)
    expected = a.copy()
    expected[index] = value

    t = FloatTensor(a.copy())
    rhs = FloatTensor(np.asarray(value, dtype=np.float64)) if isinstance(value, np.ndarray) else value
    t[index] = rhs
    np.testing.assert_array_equal(np.asarray(t), expected)
