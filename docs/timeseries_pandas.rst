From pandas
===========

Time series often arrive as a pandas ``Series`` or ``DataFrame`` indexed by
dates.
:py:meth:`TimeSeries.from_pandas <stablebear.timeseries.TimeSeries.from_pandas>`
builds a :py:class:`~stablebear.timeseries.TimeSeries` from one, and
:py:meth:`~stablebear.timeseries.TimeSeries.to_pandas` converts a time series
back. pandas is optional: stablebear does not need it otherwise.

Here six water levels are recorded every 30 minutes from 06:00 on
1 January 2023::

   import pandas as pd
   from stablebear.timeseries import TimeDelta, TimeSeries, cut_windows

   index = pd.date_range("2023-01-01T06:00", periods=6, freq="30min")
   levels = pd.Series([1.0, 3.0, 2.0, 4.0, 3.5, 2.5], index=index)

   series = TimeSeries.from_pandas(levels)
   series.epoch  # DateTime('2023-01-01T06:00:00.000000000')
   series.unit   # 'ns'

   series(TimeDelta(1, "h"))   # 2.0
   series("2023-01-01T07:15")  # 2.0

The index gives the times. A ``DatetimeIndex`` is counted from its first
date, which becomes the :py:attr:`~stablebear.timeseries.TimeSeries.epoch`
(see :doc:`timeseries_dates`). The times are stored in the resolution of the
index, nanoseconds for most pandas indexes, or in the unit passed as
``unit``, such as ``TimeSeries.from_pandas(levels, unit="min")``. A
``TimedeltaIndex`` gives durations without an epoch, and a numeric index
gives a time series without a unit, or counts of ``unit`` if it is given
(see :doc:`timeseries_units`).

The index must be sorted, without duplicates or missing times. Otherwise
``from_pandas`` raises an error that names the pandas call that finds or
fixes the problem, such as ``obj.sort_index()``. An index with a time zone
is converted to UTC and the zone is dropped, since a time series stores no
time zone.

Missing values, ``NaN`` and ``pd.NA``, are kept as ``NaN`` observations.
Call ``dropna()`` on the pandas object first to leave them out.

Several columns
---------------

A ``DataFrame`` gives one :doc:`channel <timeseries_channels>` per column,
in column order, all at the times of the index. The columns take one common
dtype, such as ``float64`` for a column of integers next to one of floats,
since all channels of a time series hold one element type. Column names are
not kept::

   frame = pd.DataFrame({"inside": [20.5, 21.0, 21.5, 21.0, 20.5, 20.0],
                         "outside": [5, 6, 8, 9, 7, 6]}, index=index)
   rooms = TimeSeries.from_pandas(frame)
   rooms.n_channels  # 2
   rooms.dtype       # float64

Windows and delay embeddings
----------------------------

:py:func:`~stablebear.timeseries.cut_windows` and
:py:func:`~stablebear.timeseries.delay_embed` take a pandas ``Series`` or
``DataFrame`` directly. They convert it with
``from_pandas`` and return the same stablebear objects as for the converted
time series, ready for
:py:func:`~stablebear.persistence.homology.compute_persistent_homology`::

   windows = cut_windows(levels, size=TimeDelta(1, "h"),
                         step=TimeDelta(30, "min"), anchor="start")
   windows.shape      # (4,)
   windows[1].domain  # from 2023-01-01T06:30 to 07:30, without the end

Back to pandas
--------------

:py:meth:`~stablebear.timeseries.TimeSeries.to_pandas` gives a pandas
``Series`` of the observations. Its index is a ``DatetimeIndex`` for a time
series with an epoch, a ``TimedeltaIndex`` of its times without one, and the
numbers for a time series without a unit. The dates and durations are at
the finer of the unit and microseconds by default, or at the
``resolution`` passed. pandas holds them to nanoseconds at most, so a time
series in a finer unit, such as picoseconds, needs ``resolution="ns"`` to
round its times to nanoseconds. Here the round trip gives back the original::

   series.to_pandas().equals(levels)  # True

A time series of several channels gives a ``DataFrame`` with columns ``0``,
``1``, and so on, indexed by all the times of all channels. Each column holds
its channel's value in effect at each time, as evaluation gives it, and
``NaN`` where the channel cannot be evaluated. Here the first channel holds
the observations at 06:00, 07:00, and 08:00, and the second those at 06:30,
07:30, and 08:30::

   first = TimeSeries.from_pandas(levels.iloc[[0, 2, 4]])
   second = TimeSeries.from_pandas(levels.iloc[[1, 3, 5]])
   TimeSeries.from_channels([first, second]).to_pandas()
   #                        0    1
   # 2023-01-01 06:00:00  1.0  NaN
   # 2023-01-01 06:30:00  1.0  3.0
   # 2023-01-01 07:00:00  2.0  3.0
   # 2023-01-01 07:30:00  2.0  4.0
   # 2023-01-01 08:00:00  3.5  4.0
   # 2023-01-01 08:30:00  NaN  2.5

Only time series of scalar numbers or booleans convert to pandas.
