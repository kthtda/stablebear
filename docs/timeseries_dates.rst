Dates and times
===============

Time series are often recorded with dates and times rather than with numbers. A
:py:class:`~stablebear.timeseries.TimeSeries` can count its timestamps from an
*epoch*, the date and time that time 0 stands for. The timestamps stay numbers
in the unit of the times, so evaluation, windows, and delay embeddings work as
on any time series, and dates can be used wherever a time series takes times.

Here three observations are recorded at 06:00, 06:30, and 08:00 on
1 January 2023. Passing their dates as a
:py:class:`~stablebear.timeseries.DateTimeTensor` counts them from the first
one, in the resolution of the dates, minutes for these, and ``to("h")``
converts them to hours::

   import datetime

   from stablebear.timeseries import (DateTime, DateTimeTensor, TimeDelta,
                                      TimeDeltaTensor, TimeSeries, Window)

   stamps = DateTimeTensor(["2023-01-01T06:00", "2023-01-01T06:30",
                            "2023-01-01T08:00"])
   series = TimeSeries([1.0, 3.0, 2.0], times=stamps).to("h")

   series.epoch  # DateTime('2023-01-01T06:00')
   series.times  # TimeDeltaTensor([0.0, 0.5, 2.0], 'h')

The differences between the dates are taken in integers and converted to
the unit once, so whole multiples of the unit, such as the 0.5 h here, are
exact. A ``datetime64`` array or a pandas ``DatetimeIndex`` gives the same
times. Pass ``epoch=`` to :py:class:`~stablebear.timeseries.TimeSeries` to
count from another date and time instead.

Epochs
------

An epoch is a :py:class:`~stablebear.timeseries.DateTime`. It is built from
a string such as ``"2023-01-01T06:00"``, a ``np.datetime64``, or a
``datetime.datetime`` or ``datetime.date``. A ``datetime`` with a timezone
is converted to UTC and its timezone dropped. An epoch keeps the resolution
of its value: a date string counts days and a ``datetime`` counts
microseconds. Two epochs are equal only at the same resolution::

   DateTime("2023-01-01") == DateTime(datetime.datetime(2023, 1, 1))  # False

Compare their ``to_datetime64()`` values to compare instants. Dates follow
the Gregorian calendar and every day has 86400 seconds: leap seconds are not
counted. Resolutions of weeks, months, and years are rejected, since months
and years have no fixed length.

An epoch is one instant. A
:py:class:`~stablebear.timeseries.DateTimeTensor` holds an array of them in
one resolution, and is what adding a
:py:class:`~stablebear.timeseries.TimeDeltaTensor` to a date gives::

   DateTime("2023-01-01") + TimeDeltaTensor([30, 90], "min")
   # DateTimeTensor(['2023-01-01T00:30', '2023-01-01T01:30'])

   DateTimeTensor(["2023-01-01T06:15", "2023-01-01T07:45"]) - series.epoch
   # TimeDeltaTensor([15, 105], 'min')

A ``DateTimeTensor`` cannot be an epoch.

.. note::

   Epochs are not recommended for times much finer than microseconds, such
   as picoseconds. Plotting rounds dates to microseconds, so such times
   collapse onto the same dates. Keep such a time series without an epoch
   instead.

All channels of a time series share one epoch, so
:py:meth:`TimeSeries.from_channels
<stablebear.timeseries.TimeSeries.from_channels>` raises if their epochs
differ. To combine channels recorded with dates, build the times of each
from its dates with the same ``epoch=``.
:py:meth:`~stablebear.timeseries.TimeSeries.with_epoch` gives a series an
epoch, or another one, without shifting any time.

Querying at dates
-----------------

A time series with an epoch can be evaluated at dates and times, given as
anything :py:class:`~stablebear.timeseries.DateTime` accepts or as an array
of them::

   series("2023-01-01T07:00")  # 3.0, the same as series(1.0, "h")
   series(DateTimeTensor(["2023-01-01T06:15",
                          "2023-01-01T07:45"]))  # [1.0, 3.0]

Durations count from the series' own time 0, and a date names an instant.
This matters for windows: a window's times start at 0, while a date is
placed in it by its :py:attr:`~stablebear.timeseries.TimeSeries.origin`, so
a window gives the value it held at that date. A date outside the domain
raises an error that names the domain as dates.

Durations
---------

The size and step of a :py:class:`~stablebear.timeseries.Window` and the
delay of a :py:class:`~stablebear.timeseries.TimeDelayEmbedding` are
durations, as on any time series with a unit. They do not need an epoch,
and are converted to the unit of each time series they are applied to::

   windows = Window(size=TimeDelta(1, "h"), step=TimeDelta(30, "min"),
                    anchor="start")(series)
   windows[1].domain  # from 2023-01-01T06:30 to 07:30, without the end

Durations can also be given as ``np.timedelta64``, ``datetime.timedelta``,
or ``pd.Timedelta``.

Dates of windows
----------------

:py:attr:`~stablebear.timeseries.TimeSeries.datetimes` gives timestamps back as
dates, at microsecond resolution, or at the unit of the series if it is finer.
The :py:attr:`~stablebear.timeseries.TimeSeries.domain` of a window with an
epoch is where it lies in the series it was cut from, as above, and the window
is evaluated at those dates::

   windows[1]("2023-01-01T07:00")  # 3.0, the same as windows[1](0.5, "h")

Microseconds reach about 290,000 years either side of 1970; nanoseconds
reach from 1678 to 2262.
