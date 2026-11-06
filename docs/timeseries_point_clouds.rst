Point clouds
============

Persistent homology takes point clouds, and a time series of numbers or
vectors traces one: each observation is a point. For a
:doc:`delay embedding <timeseries_embedding>`, the points are the embedded
vectors, whose shape reflects the dynamics of the signal.
:py:func:`~stablebear.timeseries.to_point_clouds` turns a time series into a
:py:class:`~stablebear.point_cloud.PointCloud`, and a tensor of time series,
such as :doc:`windows <timeseries_windows>`, into a
:py:class:`~stablebear.point_cloud.PointCloudTensor` of the same shape, which
:py:func:`~stablebear.persistence.homology.compute_persistent_homology`
takes directly.

The examples use the delay embedding of a sine with period 2 s from
:doc:`timeseries_embedding`, observed every 0.125 s on the domain
:math:`[0.5, 6]`::

   import numpy as np
   from stablebear.timeseries import (TimeDelta, TimeDeltaTensor, TimeSeries,
                                      cut_windows, delay_embed,
                                      to_point_clouds)

   t = np.linspace(0.0, 6.0, 49)
   signal = TimeSeries(np.sin(np.pi * t), times=TimeDeltaTensor(t, "s"))
   embedded = delay_embed(signal, dimension=2, delay=TimeDelta(0.5, "s"))

An observation of shape ``()`` gives a point of dimension 1, and one of
shape ``(m,)``, such as an embedded vector, a point of dimension
:math:`m`. The time series must have one channel; combine several with
:py:func:`~stablebear.tensor_create.stack` first. A ``float32`` time series
gives ``pcloud32`` point clouds, and time series of other numbers give
``pcloud64``.

Every observation
-----------------

By default, each observation in the domain is a point, in the order of
their times. The embedding has 45 observations from 0.5 s on::

   cloud = to_point_clouds(embedded)
   cloud.shape  # (45, 2)

Windows cut from a time series observed at irregular times hold different
numbers of observations, so their point clouds differ in size. A window
stores the value in effect at its start at time 0, and that value is a
point only if the window includes its start.

Evenly spaced times
-------------------

Point clouds of different sizes give barcodes that differ partly because
of the sampling, not the shape. To compare windows by their shape, ``n``
takes the points at ``n`` evenly spaced times
over the domain of each time series, so windows of one size give point
clouds of ``n`` points each, wherever the observations fall. Each point is
the observation in effect at its time::

   windows = cut_windows(embedded, size=TimeDelta(2, "s"),
                         step=TimeDelta(2, "s"), anchor="start")
   clouds = to_point_clouds(windows, n=8)
   clouds[0].shape  # (8, 2)

The times include each end that the domain includes. They are the evenly
spaced times over the domain with both ends closed, with ``n + 1`` times
when one end is open and ``n + 2`` when both are, the open ends left out.
On a domain from 0 to 2 s and with ``n = 3``:

======================= ========================
Domain                  Times (s)
======================= ========================
:math:`[0, 2]`          0, 1, 2
:math:`[0, 2)`          0, 2/3, 4/3
:math:`(0, 2]`          2/3, 4/3, 2
:math:`(0, 2)`          0.5, 1, 1.5
======================= ========================

The times therefore depend on which ends are closed, not only on the
length of the domain. For windows anchored at the start, which are
half-open as :math:`[s, s + \text{size})`, this makes the point clouds of
windows that follow each other form one evenly spaced grid, without a time
in two point clouds. A single point lies at the closed end, or in the middle
if both ends are open; a domain with both ends closed needs ``n`` of at
least 2.

Instead of a number of points, ``step`` gives the time between them. The
times are the start of the domain and every ``step`` after it that lies in
the domain, leaving out the start if the domain leaves it out::

   to_point_clouds(embedded, step=TimeDelta(0.5, "s")).shape  # (12, 2)

The step is a duration for a time series with a unit, or a number for one
without, as for :py:func:`~stablebear.timeseries.cut_windows`.

Given times
-----------

``times`` takes the points at the given times, in their order, the same for
every time series of a tensor. The point clouds then hold what evaluating
the time series at ``times`` gives, as in :doc:`timeseries`::

   to_point_clouds(embedded, TimeDeltaTensor([0.5, 1.0, 1.5], "s"))
   # [[0, 1], [1, 0], [0, -1]]

A time outside the domain raises a ``ValueError``. A single time series
with an epoch also takes dates and times (see :doc:`timeseries_dates`).
