Dates and times
===============

Time series are often recorded with dates and times rather than with
numbers. A :py:class:`~stablebear.timeseries.Times` can count its timestamps
from an *epoch*, the date and time that time 0 stands for. The timestamps
stay numbers in the unit of the times, so evaluation, windows, and delay
embeddings work as on any time series, and dates can be used wherever a time
series takes times.

Here three observations are recorded at 06:00, 06:30, and 08:00 on
1 January 2023. Passing their dates as a ``datetime64`` array counts them
from the first one, in hours::

   import datetime

   import numpy as np
   from stablebear.timeseries import DateTime, Times, TimeSeries, Window

   stamps = np.array(["2023-01-01T06:00", "2023-01-01T06:30",
                      "2023-01-01T08:00"], dtype="datetime64[m]")
   series = TimeSeries([1.0, 3.0, 2.0], times=Times(stamps, unit="h"))

   series.epoch  # DateTime('2023-01-01T06:00')
   series.times  # Times([0.0, 0.5, 2.0], unit='h', epoch=DateTime('2023-01-01T06:00'))

The differences between the dates are taken in integers and converted to
the unit once, so whole multiples of the unit, such as the 0.5 h here, are
exact. Pass ``epoch=`` to :py:class:`~stablebear.timeseries.Times` to count
from another date and time instead. A
:py:class:`~stablebear.timeseries.TimeSeries` built from a ``datetime64``
array directly counts in seconds.

Epochs
------

An epoch is a :py:class:`~stablebear.timeseries.DateTime`. It is built from
a string such as ``"2023-01-01T06:00"``, a ``np.datetime64``, or a
``datetime.datetime`` or ``datetime.date``. A ``datetime`` with a timezone
is converted to UTC and its timezone dropped. An epoch keeps the resolution
of its value: a date string counts days and a ``datetime`` counts
microseconds. Two epochs are equal only at the same resolution::

   DateTime("2023-01-01") == DateTime(datetime.datetime(2023, 1, 1))  # False

Compare their :py:meth:`~stablebear.timeseries.DateTime.to_datetime64`
values to compare instants. Dates follow the Gregorian calendar
and every day has 86400 seconds: leap seconds are not counted. Resolutions of
weeks, months, and years are rejected, since months and years have no fixed
length.

.. note::

   Epochs are not recommended for times much finer than microseconds, such
   as picoseconds. Dates computed from them, by
   :py:meth:`Times.to_datetime64 <stablebear.timeseries.Times.to_datetime64>`
   by default and by plotting, are rounded to microseconds, so such times
   collapse onto the same dates. Keep these times as numbers instead.

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

   series("2023-01-01T07:00")  # 3.0, the same as series(1.0)
   series(np.array(["2023-01-01T06:15", "2023-01-01T07:45"],
                   dtype="datetime64[m]"))  # [1.0, 3.0]

Numbers count from the series' own time 0, and a date names an instant.
This matters for windows: a window's times start at 0, while a date is
placed in it by its :py:attr:`~stablebear.timeseries.TimeSeries.origin`, so
a window gives the value it held at that date. A date outside the domain
raises an error that names the domain as dates.

Durations
---------

The size and step of a :py:class:`~stablebear.timeseries.Window` and the
delay of a :py:class:`~stablebear.timeseries.TimeDelayEmbedding` can be
durations, as ``np.timedelta64`` or ``datetime.timedelta``. They do not need
an epoch, and are converted to the unit of each time series they are applied
to::

   windows = Window(size=np.timedelta64(1, "h"),
                    step=np.timedelta64(30, "m"), anchor="start")(series)
   windows[1].domain  # [0, 1) in hours

Dates of windows
----------------

:py:meth:`Times.to_datetime64 <stablebear.timeseries.Times.to_datetime64>`
gives timestamps back as dates, by default at microsecond resolution, and
:py:meth:`~stablebear.timeseries.TimeSeries.bounds` gives where a window lies
in the series it was cut from, as dates with ``dates=True``::

   windows[1].bounds(dates=True)  # 06:30 to 07:30, closed at the start
   windows[1]("2023-01-01T07:00")  # 3.0, the same as windows[1](0.5)

Microseconds reach about 290,000 years either side of 1970; nanoseconds,
with ``resolution="ns"``, reach from 1678 to 2262.
