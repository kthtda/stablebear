"""Red-until-fixed regression tests for KNOWN tensor-construction bugs.

These tests come from the API bug scan (see ``bug-scan-findings.md`` at the repo
root). Each asserts the CORRECT / intended behavior, so it FAILS today and will
pass once the underlying defect is fixed. Do NOT weaken a test to make it green
-- fix the bug instead.

Area: construction (tensor constructors / ndarray + nested-list conversion).
Neither defect here is a hard crash (both repros exit 0 -- they are silent data
corruption / contract violations), so these run as ordinary in-process tests.
"""

import numpy as np
import pytest

import masspcf as mpcf


def _mk(v):
    """A trivially-valued float PCF whose first y-value encodes ``v``."""
    return mpcf.Pcf(np.array([[0.0, float(v)], [1.0, float(v)]], dtype=np.float32))


def _mki(v):
    """A trivially-valued int PCF whose first y-value encodes ``v``."""
    return mpcf.Pcf(np.array([[0, int(v)], [1, int(v)]], dtype=np.int32))


# --- Bug 0: 0-d (scalar) ndarray -> tensor drops the value ---


def test_scalar_ndarray_preserves_value():
    """0-d ndarray construction must preserve the scalar, not zero it out."""
    # BUG: Constructing a numeric/bool tensor from a 0-d (scalar) ndarray
    #      silently drops the value (returns the dtype's zero).
    # Expected: a 0-d ndarray holds exactly one element; the constructed 0-d
    #           tensor holds that scalar value (NumPy preserves scalar-array
    #           values).
    # Observed today: the single element is the dtype zero (0.0 / 0 / False),
    #                 silently, with a correct-looking Shape().

    tf = mpcf.FloatTensor(np.array(3.5, dtype=np.float64))
    assert np.asarray(tf).item() == 3.5

    ti = mpcf.IntTensor(np.array(11, dtype=np.int64))
    assert int(np.asarray(ti)) == 11

    tb = mpcf.BoolTensor(np.array(True))
    assert bool(np.asarray(tb)) is True


def test_scalar_ndarray_binding_preserves_value():
    """The C++ ndarray->tensor binding must copy the 0-d element."""
    # BUG: same defect at the binding layer (the conversion never visits the
    #      single 0-d element, so it keeps its zero default).
    # Expected: cpp.ndarray_to_tensor_64(np.array(5.0)) yields a Shape() tensor
    #           holding 5.0.
    # Observed today: Shape() holding 0.0.
    import masspcf._mpcf_cpp as cpp

    t = cpp.ndarray_to_tensor_64(np.array(5.0))
    assert np.asarray(np.array(t)).item() == 5.0


# --- Bug 1: ragged nested list silently accepted / elements misplaced ---


def test_ragged_pcf_list_raises_when_count_matches_product():
    """Ragged nested PCF lists must be rejected, even at a coincidental count."""
    # BUG: a ragged nested list whose total element count happens to equal a
    #      rectangular product is silently accepted and reshaped, scattering
    #      elements into wrong (i, j) positions.
    # Expected: ValueError reporting the ragged structure (the documented
    #           "Validates that the structure is rectangular" contract).
    # Observed today: rows of length 2, 3, 1 (total 6 == 3*2) are accepted as a
    #                 Shape(3, 2) tensor with elements at wrong positions.
    data = [[_mk(0), _mk(1)],
            [_mk(2), _mk(3), _mk(4)],
            [_mk(5)]]
    with pytest.raises(ValueError):
        mpcf.PcfTensor(data)


def test_ragged_int_pcf_list_raises_when_count_matches_product():
    """The same raggedness rejection must hold for IntPcfTensor."""
    # BUG: identical defect on the IntPcfTensor construction path.
    # Expected: ValueError for the ragged structure.
    # Observed today: accepted as Shape(3, 2).
    data = [[_mki(0), _mki(1)],
            [_mki(2), _mki(3), _mki(4)],
            [_mki(5)]]
    with pytest.raises(ValueError):
        mpcf.IntPcfTensor(data)


def test_ragged_pcf_list_rank3_raises_when_count_matches_product():
    """Rank-3 raggedness with a coincidental product must also be rejected."""
    # BUG: non-first-branch raggedness is undetected at higher ranks too.
    # Expected: ValueError; [[[a,b],[c,d]],[[e],[f,g,h]]] is ragged.
    # Observed today: accepted as Shape(2, 2, 2).
    data = [[[_mk(0), _mk(1)], [_mk(2), _mk(3)]],
            [[_mk(4)], [_mk(5), _mk(6), _mk(7)]]]
    with pytest.raises(ValueError):
        mpcf.PcfTensor(data)
