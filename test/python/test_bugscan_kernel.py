"""Red-until-fixed regression tests for confirmed kernel-area bugs.

These tests document a *known, currently-unfixed* defect found by the API bug
scan (see ``bug-scan-findings.md`` at the repo root for the catalogue and
root-cause hints). Each test asserts the CORRECT / intended behavior, so it
FAILS today and will pass once the bug is fixed. Do **not** weaken a test to
make it green -- fix the bug instead.

The defect here is a hard SIGSEGV in the C++ pairwise-integration backend, so
the crashing ops are exercised ONLY through the subprocess helpers in
``_bugscan_support`` -- never directly in-process -- to avoid aborting the whole
pytest session.
"""

from _bugscan_support import assert_ok


def test_l2_kernel_and_pdist_on_reversed_view_compute_correct_values():
    """l2_kernel/pdist on a negative-step PCF view must not segfault.

    BUG: l2_kernel (and pdist) segfault on a negative-step PCF tensor view
    because Tensor1dValueIterator stores its stride as unsigned size_t, so a
    negative stride wraps to a huge value and the pairwise-integration task
    iterates out of bounds.

    Expected: the reversed view yields the same kernel/distance as a
    materialized contiguous copy (it currently does for positive-step strided
    views and for a materialized copy), namely
        l2_kernel: [[18, 12, 6], [12, 8, 4], [6, 4, 2]]
        pdist:     [[0, 2, 4], [2, 0, 2], [4, 2, 0]]
    or, at worst, a clean Python exception -- not a memory-unsafe crash.

    Observed today: mpcf.l2_kernel(X[::-1]) and mpcf.pdist(X[::-1]) both die
    with SIGSEGV (process exit code 139, core dumped). Confirmed for pcf64 and
    pcf32; both ops share the same CpuPairwiseIntegrationTask path.
    """
    # Exercised in a subprocess: a direct call segfaults and would abort pytest.
    assert_ok(
        """
        for dtype in (mpcf.pcf64, mpcf.pcf32):
            npdt = np.float64 if dtype is mpcf.pcf64 else np.float32

            X = mpcf.zeros((3,), dtype=dtype)
            for i in range(3):
                X[i] = mpcf.Pcf(
                    np.array([[0.0, float(i + 1)], [2.0, 0.0]], dtype=npdt))

            rev = X[::-1]

            # Sanity: the reversed view itself is fine -- element access works.
            vals = [rev[k].to_numpy()[0, 1] for k in range(3)]
            assert vals == [3.0, 2.0, 1.0], vals

            # BUG: this call hard-crashes today.
            K = np.asarray(mpcf.l2_kernel(rev).to_dense())
            expected_K = np.array(
                [[18.0, 12.0, 6.0], [12.0, 8.0, 4.0], [6.0, 4.0, 2.0]])
            assert np.allclose(K, expected_K), (dtype, K)

            # Same root cause: pdist also crashes today.
            D = np.asarray(mpcf.pdist(rev).to_dense())
            expected_D = np.array(
                [[0.0, 2.0, 4.0], [2.0, 0.0, 2.0], [4.0, 2.0, 0.0]])
            assert np.allclose(D, expected_D), (dtype, D)

        print("BUGSCAN_OK")
        """
    )
