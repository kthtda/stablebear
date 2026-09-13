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

All births are zero; positive effective merge heights become finite death times.
Zero-length intervals are omitted, and ``reduced=True`` removes the essential
infinite interval. Cluster relationships and tie-breaking results are preserved.
Only the example's clustering step requires SciPy; conversion and validation run in C++ with the Python GIL released.

Input must be an ``(n - 1, 4)`` NumPy array with float32 or float64 precision,
which is preserved in the output. The input is not modified. Empty ``(0, 4)``
input represents a single observation, producing one infinite interval (or an
empty barcode in reduced mode). Invalid cluster references or counts, non-finite
entries, and negative heights raise ``ValueError``.

Inversions, such as those from centroid or median linkage, are handled by
delaying each parent merge until both child clusters exist. Its effective
height is the maximum of its supplied height and its children's effective
heights. For example, child merges at 3.0 and 3.5 followed by a parent merge
at 3.25 produce finite bars ending at 3.0, 3.5, and 3.5. Heights are not
sorted, and unrelated branches do not delay one another.

Conversion emits one ``UserWarning`` when input merge heights are non-monotone:
a row's height is lower than the preceding row's height. This includes decreases
on independent branches, even when no parent height needs adjustment. Equal
heights are allowed and do not trigger a warning.
