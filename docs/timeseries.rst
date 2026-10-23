Time series
===========

A time series records observations at known times. Each observation can be a
number, an array, or an element such as a point cloud. A
:py:class:`~stablebear.timeseries.TimeSeries` is built from the observations
and one timestamp per observation; plain timestamp arrays are in seconds, and
:py:class:`~stablebear.timeseries.Times` gives them another unit::

   from stablebear.timeseries import Times, TimeSeries

   series = TimeSeries([1.0, 3.0, 2.0, 4.0], times=[0.0, 1.0, 1.5, 3.5])

   times_ms = Times([0.0, 1000.0, 1500.0, 3500.0], unit="ms")
   series_ms = TimeSeries([1.0, 3.0, 2.0, 4.0], times=times_ms)

Both time series hold the same observations at the same times; ``series_ms``
reads query times in milliseconds, so ``series_ms(1500.0)`` equals
``series(1.5)``.

This page covers evaluation and units, which every other operation on a
time series builds on. The pages listed at the end cover channels, delay
embeddings, and windows.

Evaluation
----------

For observations :math:`x_0, \dots, x_{n-1}` at timestamps
:math:`t_0 < \dots < t_{n-1}`,
:py:meth:`~stablebear.timeseries.TimeSeries.evaluate` computes

.. math::

   x(t) = x_i \quad \text{for } t_i \le t < t_{i+1},
   \qquad x(t_{n-1}) = x_{n-1},

on the domain :math:`[t_0, t_{n-1}]`. The figure shows ``series`` from the
example above as a step function on its domain :math:`[0, 3.5]`. The filled
circles are the observations, and an open circle marks the right end of a
step, which belongs to the next step. The squares are queries at ``t = 0.5``,
``1.5``, and ``2.75``.

.. image:: _static/timeseries_evaluation_light.png
   :width: 80%
   :class: only-light
   :alt: A time series as a step function, evaluated at t = 0.5, 1.5, and 2.75

.. image:: _static/timeseries_evaluation_dark.png
   :width: 80%
   :class: only-dark
   :alt: A time series as a step function, evaluated at t = 0.5, 1.5, and 2.75

.. dropdown:: Show plotting code
   :color: secondary

   .. literalinclude:: _static/gen_timeseries_snap_fig.py
      :pyobject: plot_timeseries_evaluation
      :language: python

Each query returns the value of the step it falls on. A query past the last
timestamp is outside the domain and raises an error::

   series(0.5)   # 1.0
   series(1.5)   # 2.0
   series(2.75)  # 2.0
   series(4.0)   # ValueError: Time 4 is outside the domain [0, 3.5]

A query at a timestamp gets the observation recorded there, as at
``t = 1.5``.

Units
-----

A :py:class:`~stablebear.timeseries.Times` holds timestamps in one physical
unit, from attoseconds (``"as"``) to days (``"d"``); :doc:`timeseries_units`
lists them all.
:py:meth:`Times.to <stablebear.timeseries.Times.to>` converts them by an
exact integer factor::

   times = Times([0.0, 250.0, 500.0], unit="ms")
   times.to("s")  # Times([0.0, 0.25, 0.5], unit='s')

A time series reads query times, window sizes, and delays in its unit, which
:py:meth:`TimeSeries.to <stablebear.timeseries.TimeSeries.to>` changes
without converting the stored timestamps. A unit passed as the second
argument of :py:meth:`~stablebear.timeseries.TimeSeries.evaluate` applies to
that query only::

   series(1500.0, "ms")  # 2.0, the same as series(1.5)

Working with time series
------------------------

The following pages build on evaluation. Channels combine signals with
different timestamps and units into one series. A delay embedding turns a
signal into points that trace its dynamics, and windows cut a time series into
pieces that can be compared with each other.
Snapping covers query times such as ``0.3 - 0.1`` that land a rounding error
below the timestamp they mean.

.. toctree::
   :maxdepth: 1

   timeseries_units
   timeseries_channels
   timeseries_embedding
   timeseries_windows
   timeseries_snapping
