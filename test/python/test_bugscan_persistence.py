"""Red-until-fixed regression tests for confirmed persistence-area bugs.

These tests were authored from a broad API bug scan (see ``bug-scan-findings.md``
at the repo root and ``/tmp/bugscan/persistence.json``). Each test asserts the
CORRECT / intended behavior taken from each bug's ``expected`` field, so it FAILS
TODAY (red) against the installed package and will pass once the underlying bug
is fixed. Do NOT weaken a test to make it green -- fix the bug instead.

All four defects in this area were confirmed to exit cleanly (no signal death /
segfault) when reproduced standalone, so they are written as ordinary in-process
tests: an uncaught exception or failed assert here fails cleanly without aborting
the pytest session.
"""

import inspect
import io
from contextlib import redirect_stderr

import numpy as np
import pytest

import masspcf as mpcf
import masspcf.persistence as mpers


def test_malformed_pointcloud_raises_instead_of_silent_empty_barcodes():
    """Bug 26: rank!=2 point cloud silently yields empty barcodes (swallowed throw)."""
    # BUG: Malformed point cloud (rank != 2) silently yields empty barcodes; the
    #      deliberate C++ runtime_error is swallowed because the task future's
    #      .get() is never called.
    # Expected: compute_persistent_homology must surface a clean error (the C++
    #           backend deliberately throws std::runtime_error("... has
    #           unexpected shape ... (should be (m, n))"), which pybind11 maps to
    #           a Python RuntimeError) rather than silently returning all-empty
    #           barcodes -- empty output is indistinguishable from a genuinely
    #           trivial result and is undetectable data corruption.
    # Observed today: returns a correctly-shaped BarcodeTensor whose every
    #                 barcode is empty (H0=[], H1=[]) with no exception, exit 0.

    # First confirm a VALID control input computes a non-trivial H0 in this run,
    # so the empty result below is specific to the malformed input.
    valid = mpcf.zeros((1,), dtype=mpcf.pcloud64)
    valid[0] = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    ok = mpers.compute_persistent_homology(valid, max_dim=1, verbose=False)
    assert len(np.asarray(ok[0])) > 0, "control (3,2) cloud should produce H0 bars"

    # rank-1 cloud (4,) passes the zero-dim guard but is not (m, n): must error.
    bad = mpcf.zeros((1,), dtype=mpcf.pcloud64)
    bad[0] = mpcf.FloatTensor(np.array([1.0, 2.0, 3.0, 4.0]))
    with pytest.raises(Exception):
        mpers.compute_persistent_homology(bad, max_dim=1, verbose=False)

    # Same defect via raw numpy arrays of rank 1 and rank 3.
    with pytest.raises(Exception):
        mpers.compute_persistent_homology(
            np.array([0.0, 5.0, 10.0]), max_dim=1, verbose=False)
    with pytest.raises(Exception):
        mpers.compute_persistent_homology(
            np.zeros((2, 3, 2)), max_dim=1, verbose=False)


def test_compute_persistent_homology_preserves_length1_batch_dim():
    """Bug 27: genuine (1,) tensor input is squeezed to (max_dim+1,) not (1, max_dim+1)."""
    # BUG: compute_persistent_homology squeezes the batch dimension for a genuine
    #      length-1 (1,) PointCloudTensor/DistanceMatrixTensor.
    # Expected: per the docstring "For an input tensor of shape (d_1, ..., d_n),
    #           the output has shape (d_1, ..., d_n, max_dim+1)", a genuine
    #           (1,)-shaped batch tensor must yield output shape (1, max_dim+1)
    #           and bcs[0, 0] must index the first element's H0.
    # Observed today: output shape (max_dim+1,) = (2,); bcs[0, 0] -> IndexError.

    # Point-cloud path.
    X = mpcf.zeros((1,), dtype=mpcf.pcloud64)
    X[0] = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    bcs = mpers.compute_persistent_homology(X, max_dim=1, verbose=False)
    assert bcs.shape == (1, 2), (
        f"genuine (1,) pcloud batch should give shape (1, 2), got {bcs.shape}")
    # Natural batch indexing must work (this raises IndexError today).
    bcs[0, 0]

    # Distance-matrix path: same defect.
    d = mpcf.DistanceMatrix(3, dtype=mpcf.float64)
    d[0, 1] = 1.0
    d[0, 2] = 1.0
    d[1, 2] = np.sqrt(2.0)
    dmt = mpcf.zeros((1,), dtype=mpcf.distmat64)
    dmt[0] = d
    bcs_dm = mpers.compute_persistent_homology(dmt, max_dim=1, verbose=False)
    assert bcs_dm.shape == (1, 2), (
        f"genuine (1,) distmat batch should give shape (1, 2), got {bcs_dm.shape}")
    bcs_dm[0, 0]


def test_barcode_summaries_preserve_leading_dim1_shape():
    """Bug 28: summary funcs drop the leading dim for any 2-D (1, N) BarcodeTensor."""
    # BUG: barcode_to_stable_rank/betti_curve/accumulated_persistence drop the
    #      leading dimension for any 2-D BarcodeTensor with leading dim 1
    #      (e.g. (1,2) -> (2,)).
    # Expected: per each function's docstring ("a PcfTensor with the same shape
    #           as the input"), input (1, 2) must yield output (1, 2) and (1, 3)
    #           must yield (1, 3); out[0, 1] must succeed.
    # Observed today: output is squeezed to (2,) / (3,); out[0, 1] -> IndexError.

    bcs = mpcf.zeros((1, 2), dtype=mpcf.barcode64)
    bcs[0, 0] = mpers.Barcode(np.array([[0.0, 1.0]]))
    bcs[0, 1] = mpers.Barcode(np.array([[0.0, 2.0]]))

    sr = mpers.barcode_to_stable_rank(bcs)
    assert sr.shape == (1, 2), f"stable_rank should preserve (1, 2), got {sr.shape}"
    sr[0, 1]  # must not raise

    betti = mpers.barcode_to_betti_curve(bcs)
    assert betti.shape == (1, 2), f"betti_curve should preserve (1, 2), got {betti.shape}"
    betti[0, 1]

    apf = mpers.barcode_to_accumulated_persistence(bcs, verbose=False)
    assert apf.shape == (1, 2), (
        f"accumulated_persistence should preserve (1, 2), got {apf.shape}")
    apf[0, 1]

    # Also pin (1, 3) -> (1, 3) for stable_rank to confirm it is not special to N=2.
    bcs3 = mpcf.zeros((1, 3), dtype=mpcf.barcode64)
    for j in range(3):
        bcs3[0, j] = mpers.Barcode(np.array([[0.0, 1.0 + j]]))
    sr3 = mpers.barcode_to_stable_rank(bcs3)
    assert sr3.shape == (1, 3), f"stable_rank should preserve (1, 3), got {sr3.shape}"


def test_accumulated_persistence_verbose_defaults_to_false():
    """Bug 29: barcode_to_accumulated_persistence defaults verbose=True (prints tqdm bar)."""
    # BUG: barcode_to_accumulated_persistence defaults verbose=True, contradicting
    #      its docstring ("by default False"), the two sibling summary functions,
    #      and the Sphinx docs.
    # Expected: the verbose default must be False (silent by default), matching the
    #           docstring, both siblings, and docs/persistence.rst.
    # Observed today: signature default is True and a tqdm progress bar is emitted
    #                 to stderr on a default (no-verbose-arg) call.

    ap_default = inspect.signature(
        mpers.barcode_to_accumulated_persistence).parameters["verbose"].default
    sr_default = inspect.signature(
        mpers.barcode_to_stable_rank).parameters["verbose"].default
    bc_default = inspect.signature(
        mpers.barcode_to_betti_curve).parameters["verbose"].default

    # Siblings are silent by default (sanity anchors); AP must match them.
    assert sr_default is False
    assert bc_default is False
    assert ap_default is False, (
        "barcode_to_accumulated_persistence verbose default should be False "
        f"to match its docstring and siblings, got {ap_default!r}")

    # And a default call must not emit a tqdm progress bar to stderr.
    bcs = mpcf.zeros((3,), dtype=mpcf.barcode64)
    for i in range(3):
        bcs[i] = mpers.Barcode(np.array([[0.0, 1.0], [0.0, 2.0]]))
    buf = io.StringIO()
    with redirect_stderr(buf):
        mpers.barcode_to_accumulated_persistence(bcs)
    assert "Converting barcodes" not in buf.getvalue(), (
        "default (no verbose arg) call should be silent, but printed a progress "
        f"bar: {buf.getvalue()[:120]!r}")
