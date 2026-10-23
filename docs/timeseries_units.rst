Time units
==========

:py:class:`~stablebear.timeseries.Times` and
:py:class:`~stablebear.timeseries.TimeSeries` accept the following units; see
:doc:`timeseries` for how a time series uses its unit. Names are case-sensitive.
:py:attr:`Times.unit <stablebear.timeseries.Times.unit>` reports the short
name, whichever spelling was passed.

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
     - ``"day"``, ``"days"``

The unit ``"µs"`` is accepted with either the micro sign (U+00B5) or the
Greek letter mu (U+03BC).
