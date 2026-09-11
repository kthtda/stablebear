=================================
Converting SciPy linkage matrices
=================================

Use ``linkage_to_barcode`` to turn an existing hierarchical clustering result
into a single ``Barcode``::

   from scipy.cluster.hierarchy import linkage
   from stablebear.persistence import linkage_to_barcode

   Z = linkage([[0.0], [1.0], [4.0]], method="single")
   bc = linkage_to_barcode(Z)
   # Intervals: [0, 1), [0, 3), [0, inf).
   reduced_bc = linkage_to_barcode(Z, reduced=True)
   # Intervals: [0, 1), [0, 3).

All births are zero; positive merge heights become finite death times.
Zero-length intervals are omitted, and ``reduced=True`` removes the essential
infinite interval. SciPy's merge heights and tie-breaking results are preserved.
Only the example's clustering step requires SciPy; conversion and validation run in C++ with the Python GIL released.

Input must be an ``(n - 1, 4)`` NumPy array with float32 or float64 precision,
which is preserved in the output. The input is not modified. Empty ``(0, 4)``
input represents a single observation, producing one infinite interval (or an
empty barcode in reduced mode). Invalid cluster references or counts, non-finite
entries, negative heights, and decreasing heights raise ``ValueError``.
In particular, centroid/median linkage results with height inversions are
rejected; heights are never silently sorted or repaired.
