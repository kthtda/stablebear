Windows
=======

A window restricts a series to an interval of time, so that pieces of one
signal can be compared with each other. :py:class:`~stablebear.timeseries.Window`
cuts a series into windows of a fixed size at a fixed step, ending where
the series ends or, from the start, starting where it starts. Each window is
a series whose times start at 0.

Here the windows cut the :doc:`delay embedding <timeseries_embedding>` of a
sine with period 2 s, whose domain is :math:`[0.5, 5.9]`, into pieces of
2 s, one every second, the last ending at 5.9 s::

   import numpy as np
   from stablebear.persistence import compute_persistent_homology
   from stablebear.timeseries import TimeDelayEmbedding, TimeSeries, Window

   t = np.arange(0.0, 6.0, 0.1)
   signal = TimeSeries(np.sin(np.pi * t), times=t)
   embedded = TimeDelayEmbedding(dimension=2, delay=0.5)(signal)

   windows, bounds = Window(size=2.0, step=1.0)(embedded, return_bounds=True)
   bounds  # [[0.9, 2.9], [1.9, 3.9], [2.9, 4.9], [3.9, 5.9]], up to rounding

Each window is :math:`(a, a + 2]`, so it can be evaluated on a grid of 40
times in :math:`(0, 2]`. Evaluating every window there with
:py:meth:`TimeSeriesTensor.evaluate
<stablebear.timeseries.series.TimeSeriesTensor.evaluate>` gives a tensor of
shape ``(4, 40, 2)``: one point cloud of 40 points in the plane per window.
The persistence of each point cloud measures the loop::

   points = windows(np.arange(1, 41) * 2.0 / 40)
   barcodes = compute_persistent_homology(points, max_dim=1)

Each window holds one full period of the circle, so every window gives the
same longest H1 bar, of length about 1.47. A change in the dynamics of the
signal would show up as a change in the bars from one window to the next.
On a series with several :doc:`channels <timeseries_channels>`, each channel
is windowed separately.

Windows from the start
----------------------

By default the last window ends where the series ends, and time at the
start that no full window covers is left out. With ``anchor="start"``, the
first window starts where the series starts instead, and the time left out
is at the end::

   series = TimeSeries([0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
                       times=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

   _, bounds = Window(size=2.0)(series, return_bounds=True)
   bounds  # [[1.0, 3.0], [3.0, 5.0]]

   _, bounds = Window(size=2.0, anchor="start")(series, return_bounds=True)
   bounds  # [[0.0, 2.0], [2.0, 4.0]]

Either way the windows come in chronological order. By default each window
is closed at its anchor end, :math:`(a, a + \text{size}]` from the end and
:math:`[a, a + \text{size})` from the start, so the observation at 5 s is in
the last window, and the one at 0 s is in the first window from the start.
With ``partial=True``, the time left out becomes a shorter window, at the
start of the series or, with ``anchor="start"``, at its end.
