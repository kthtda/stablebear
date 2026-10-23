Time series
===========

A time series records observations at known times. Each observation can be a
number, an array, or an element such as a point cloud. A
:py:class:`~stablebear.timeseries.TimeSeries` is built from the observations
and one timestamp per observation; plain timestamp arrays are in seconds, and
:py:class:`~stablebear.timeseries.Times` gives them another unit::

   from stablebear.timeseries import Times, TimeSeries

   series = TimeSeries([1.0, 3.0, 2.0, 4.0], times=[0.0, 1.0, 1.5, 3.5])
   series_ms = TimeSeries([1.0, 3.0, 2.0, 4.0],
                          times=Times([0.0, 1000.0, 1500.0, 3500.0],
                                      unit="ms"))

Both series hold the same observations at the same times; ``series_ms``
reads query times in milliseconds, so ``series_ms(1500.0)`` equals
``series(1.5)``.

This page covers evaluation and units, which every other operation on a
series builds on. The pages listed at the end cover channels, operations on
the observations, delay embeddings, and windows.

Evaluation
----------

For observations :math:`x_0, \dots, x_{n-1}` at timestamps
:math:`t_0 < \dots < t_{n-1}`,
:py:meth:`~stablebear.timeseries.TimeSeries.evaluate` computes

.. math::

   x(t) = x_i \quad \text{for } t_i \le t < t_{i+1},
   \qquad x(t_{n-1}) = x_{n-1},

on the domain :math:`[t_0, t_{n-1}]`::

   series(0.5)   # 1.0
   series(1.5)   # 2.0
   series(2.75)  # 2.0
   series(4.0)   # ValueError: Time 4 is outside the domain [0, 3.5]

The figure shows the series as a step function. The circles are the
observations, and the squares are the three queries above.

.. image:: _static/timeseries_evaluation_light.png
   :width: 80%
   :class: only-light
   :alt: A series as a step function, evaluated at t = 0.5, 1.5, and 2.75

.. image:: _static/timeseries_evaluation_dark.png
   :width: 80%
   :class: only-dark
   :alt: A series as a step function, evaluated at t = 0.5, 1.5, and 2.75

.. dropdown:: Show plotting code
   :color: secondary

   .. literalinclude:: _static/gen_timeseries_snap_fig.py
      :pyobject: plot_timeseries_evaluation
      :language: python

A query at a timestamp gets the observation recorded there, as at
``t = 1.5``.

Queries computed in floating point
----------------------------------

Query times are often computed, for example as ``0.3 - 0.1``, and rounding
can put them just below the timestamp they mean. Evaluation snaps such
queries forward to that timestamp, so on a series with a timestamp at 0.2,
the query ``0.3 - 0.1`` returns the observation recorded there rather than
the one before it. Only queries within a rounding error of a timestamp
snap; the defaults fit most data. The ``snap_tolerance`` and ``snap_eps``
arguments of :py:class:`~stablebear.timeseries.TimeSeries` set the limits,
and :doc:`timeseries_snapping` explains them in detail.

Units
-----

A :py:class:`~stablebear.timeseries.Times` holds timestamps in one physical
unit, from attoseconds (``"as"``) to days (``"d"``).
:py:meth:`Times.to <stablebear.timeseries.Times.to>` converts them by an
exact integer factor::

   times = Times([0.0, 250.0, 500.0], unit="ms")
   times.to("s")  # Times([0.0, 0.25, 0.5], unit='s')

A series reads query times, window sizes, and delays in its unit, which
:py:meth:`TimeSeries.to <stablebear.timeseries.TimeSeries.to>` changes
without converting the stored timestamps.

Working with time series
------------------------

The following pages build on evaluation. Channels combine signals with
different timestamps and units into one series. Element-wise operations,
such as persistent homology, apply to every observation and keep the times.
A delay embedding turns a signal into points that trace its dynamics, and
windows cut a series into pieces that can be compared with each other.

.. toctree::
   :maxdepth: 1

   timeseries_channels
   timeseries_operations
   timeseries_embedding
   timeseries_windows
   timeseries_snapping
