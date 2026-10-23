Windows
=======

A window restricts a series to an interval of time, so that pieces of one
signal can be compared with each other. :py:class:`~stablebear.timeseries.Window`
cuts a series into windows of a fixed size at a fixed step, starting where
the series starts. Each window is a series whose times start at 0.

Here the windows cut the :doc:`delay embedding <timeseries_embedding>` of a
sine with period 2 s, whose domain is :math:`[0.5, 5.9]`, into pieces of
2 s, one every second::

   import numpy as np
   from stablebear.persistence import compute_persistent_homology
   from stablebear.timeseries import TimeDelayEmbedding, TimeSeries, Window

   t = np.arange(0.0, 6.0, 0.1)
   signal = TimeSeries(np.sin(np.pi * t), times=t)
   embedded = TimeDelayEmbedding(dimension=2, delay=0.5)(signal)

   windows, bounds = Window(size=2.0, step=1.0)(embedded, return_bounds=True)
   bounds  # [[0.5, 2.5], [1.5, 3.5], [2.5, 4.5], [3.5, 5.5]]

Evaluating every window on a grid of 40 times in :math:`[0, 2)` with
:py:meth:`TimeSeriesTensor.evaluate
<stablebear.timeseries.series.TimeSeriesTensor.evaluate>` gives a tensor of
shape ``(4, 40, 2)``: one point cloud of 40 points in the plane per window.
The persistence of each point cloud measures the loop::

   points = windows(np.arange(40) * 2.0 / 40)
   barcodes = compute_persistent_homology(points, max_dim=1)

Each window holds one full period of the circle, so every window gives the
same longest H1 bar, of length about 1.47. A change in the dynamics of the
signal would show up as a change in the bars from one window to the next.
On a series with several :doc:`channels <timeseries_channels>`, each channel
is windowed separately.
