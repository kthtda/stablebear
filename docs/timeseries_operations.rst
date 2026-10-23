Operations on time series
=========================

Element-wise operations accept a
:py:class:`~stablebear.timeseries.TimeSeries` wherever they accept a tensor
of the same elements, and keep its times. The result is again a series, and
evaluating it at a time gives the result of the operation on the
observation in effect at that time; see :doc:`timeseries`.

Persistence through time
------------------------

Consider a regular 12-gon whose circumradius grows from 1 to 3 over two
seconds::

   import numpy as np
   import stablebear as sb
   from stablebear.persistence import (
       barcode_to_stable_rank,
       compute_persistent_homology,
   )
   from stablebear.timeseries import TimeSeries

   angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
   polygon = np.column_stack([np.cos(angles), np.sin(angles)])
   point_clouds = sb.PointCloudTensor([r * polygon for r in (1.0, 2.0, 3.0)])
   snapshots = TimeSeries(point_clouds, times=[0.0, 1.0, 2.0])

:py:func:`~stablebear.persistence.homology.compute_persistent_homology`
gives a series of barcodes with the same times, one barcode per homology
degree at each time, and
:py:func:`~stablebear.persistence.barcode_summary.barcode_to_stable_rank`
turns those into a series of stable ranks::

   barcodes = compute_persistent_homology(snapshots, max_dim=1)
   barcodes.times   # Times([0.0, 1.0, 2.0], unit='s')
   barcodes.shape   # (2,), degrees 0 and 1
   ranks = barcode_to_stable_rank(barcodes)

The single H1 bar of the polygon is born when neighboring vertices join, at
the side length :math:`2r\sin(\pi/12) \approx 0.52r`, and dies at
:math:`2r\sin(\pi/3) \approx 1.73r`, so its length grows in proportion to
the circumradius :math:`r`: about 1.21, 2.43, and 3.64. Evaluating the
series of stable ranks between observations gives the most recent ones,
here those at 1 s, where the H1 bar is longer than 2 but not 2.5::

   h1 = ranks(1.5)[1]
   h1(np.array([0.0, 2.0, 2.5]))  # [1.0, 1.0, 0.0]
