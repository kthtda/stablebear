Time units
==========

A :py:class:`~stablebear.timeseries.TimeSeries` stores its timestamps as
float64 numbers of its unit, counted from its time 0. The unit therefore
sets which times are exact: whole multiples of the unit, and fractions such
as 0.5 whose binary representation is exact, are stored exactly, while a
time such as 0.1 s is the nearest float64 number to 0.1. Every channel of a
time series and every window cut from it store their times in the same unit.

Durations are converted into the unit of a time series when they are used,
and :py:meth:`TimeSeries.to <stablebear.timeseries.TimeSeries.to>` converts
the stored timestamps into another unit. Each conversion multiplies or
divides by an exact integer factor, such as 1000 between seconds and
milliseconds, so it rounds each time once. Choose a unit in which the
times of interest are whole numbers, such as milliseconds for times
recorded to the millisecond, to keep them exact.

:py:class:`~stablebear.timeseries.TimeDelta`,
:py:class:`~stablebear.timeseries.TimeDeltaTensor`, and
:py:class:`~stablebear.timeseries.TimeSeries` accept the following units.
Names are case-sensitive.
:py:attr:`TimeSeries.unit <stablebear.timeseries.TimeSeries.unit>` reports
the short name, whichever spelling was passed. A time series built from
plain numbers has no unit (see :doc:`timeseries`).

.. list-table::
   :header-rows: 1
   :widths: 10 20 25 45

   * - Unit
     - Name
     - Length
     - Other spellings
   * - ``"as"``
     - attosecond
     - :math:`10^{-18}` s
     - ``"attosecond"``, ``"attoseconds"``
   * - ``"fs"``
     - femtosecond
     - :math:`10^{-15}` s
     - ``"femtosecond"``, ``"femtoseconds"``
   * - ``"ps"``
     - picosecond
     - :math:`10^{-12}` s
     - ``"picosecond"``, ``"picoseconds"``
   * - ``"ns"``
     - nanosecond
     - :math:`10^{-9}` s
     - ``"nanosecond"``, ``"nanoseconds"``
   * - ``"us"``
     - microsecond
     - :math:`10^{-6}` s
     - ``"microsecond"``, ``"microseconds"``, ``"usec"``, ``"µs"``
   * - ``"ms"``
     - millisecond
     - :math:`10^{-3}` s
     - ``"millisecond"``, ``"milliseconds"``, ``"msec"``
   * - ``"s"``
     - second
     - 1 s
     - ``"second"``, ``"seconds"``, ``"sec"``
   * - ``"min"``
     - minute
     - 60 s
     - ``"minute"``, ``"minutes"``
   * - ``"h"``
     - hour
     - 3600 s
     - ``"hour"``, ``"hours"``, ``"hr"``
   * - ``"d"``
     - day
     - 86400 s
     - ``"day"``, ``"days"``, ``"D"``

The unit ``"µs"`` is accepted with either the micro sign (U+00B5) or the
Greek letter mu (U+03BC).
