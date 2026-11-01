# Durations and dates instead of numbers for time series: spec

Issue: [#258](https://github.com/kthtda/stablebear/issues/258), which this
grew out of

Plan: [pandas-timeseries-plan.md](pandas-timeseries-plan.md), phases C to G

## Goal

A plain number such as `series(3.5)` or `Window(size=3.0)` means
something only together with the unit of the series, which the reader
cannot see. The time series API should take and give times as durations
and dates instead, so that every time says what it measures. Time series
are in beta, so the old forms are removed rather than deprecated.

## Principle

- A **duration** is a `TimeDelta`, `np.timedelta64`, `datetime.timedelta`,
  or `pd.Timedelta`.
- A **date** is a `DateTime`, or a value it accepts.
- A **time** of a series is a duration measured from its time 0. For a
  series with an epoch, it can also be a date. A window's time 0 is its
  start.
- A series without an epoch gives its times as durations, and a series with
  an epoch gives them as dates.
- The `unit` of a series remains, but only as its storage: its times are
  floats in that unit. It sets their precision and the unit of a plot
  axis. No number in the API is read in it, except when a `Times` is built
  from numbers.

## `TimeDelta`

`TimeDelta(value, unit=None)` is a count of a fixed-length unit. The count
is an integer, as `np.timedelta64` and `DateTime` store theirs, or a float,
as a time series stores its times.

- `value` is an `np.timedelta64`, `datetime.timedelta`, `pd.Timedelta`, or
  `TimeDelta`, kept at its resolution with an integer count. With `unit`,
  `value` is a number of `unit`, as in `TimeDelta(1.5, "h")`; a float that
  is a whole number becomes an integer.
- It has `count`, `unit`, `to_timedelta64(resolution=None)`, `==`, `hash`,
  `repr`, and pickling, as `DateTime` does. Equality compares count and
  unit; compare `to_timedelta64()` values to compare lengths.
  `to_timedelta64` rounds to `resolution`, which defaults to the unit for
  an integer count and to the finer of the unit and microseconds for a
  float.
- Arithmetic gives `TimeDelta` and `DateTime` results:
  - `TimeDelta` plus or minus `TimeDelta`, negation, multiplication by a
    number, and comparisons, in the finer of the operands' units,
    converted by exact factors;
  - `DateTime` plus or minus `TimeDelta`, and `DateTime` minus `DateTime`;
    a float count is first rounded as `to_timedelta64` rounds it;
  - comparisons of `DateTime`.
- It rejects `NaT`, counts that are not finite, and weeks, months, and
  years.
- The count can also be an array of counts in one unit, as in
  `TimeDelta(sb.FloatTensor([[30, 90], [240, 60]]), "min")`. With `unit`,
  `value` is then a `FloatTensor`, `IntTensor`, `ndarray`, or nested list
  of numbers, kept as a `FloatTensor`, or an `IntTensor` for integers so
  that their counts stay exact; a `timedelta64` array is kept likewise at
  its resolution. An array `TimeDelta` works as a scalar one does,
  elementwise:
  - it has `shape`, `ndim`, `len`, iteration, and indexing, which gives a
    scalar `TimeDelta` for one element and an array one otherwise;
  - `count` is the tensor; `to_timedelta64` gives a `timedelta64` array;
  - arithmetic, by an exact factor per element, gives an array
    `TimeDelta`, and `<`, `<=`, `>`, `>=`, `==`, and `!=` give a
    `BoolTensor`, as tensors do; `==` compares count and unit, as for a
    scalar. `array_equal` tells whether two are equal as a whole;
  - it has `repr` and pickling but no hash.
- A `TimeDelta` given to `Window`, `TimeDelayEmbedding`, or `evaluate` is
  converted to the series' unit by an exact factor, without rounding.
  `evaluate` and `Times` take an array `TimeDelta` as an array of
  durations.
- `TimeDelta` is Python only; nothing in C++ stores a duration.

## `DateTime` arrays

`DateTime` can also hold an array of instants in one unit, as `TimeDelta`
holds an array of counts.

- `value` is then a `datetime64` array, a pandas `DatetimeIndex` or
  `Series` of dates, or a sequence of values a scalar `DateTime` accepts,
  such as strings. The instants are kept at their resolution, or the finest
  of a sequence, as an `IntTensor` of counts after 1970-01-01T00:00. A
  timezone is converted to UTC and dropped, as for a scalar.
- An array `DateTime` works as a scalar one does, elementwise:
  - it has `shape`, `ndim`, `len`, iteration, and indexing, which gives a
    scalar `DateTime` for one element and an array one otherwise;
  - `count` is the tensor; `to_datetime64` gives a `datetime64` array;
  - `<`, `<=`, `>`, `>=`, `==`, and `!=` give a `BoolTensor`; `==`
    compares count and unit, as for a scalar. `array_equal` tells whether
    two are equal as a whole;
  - it has `repr` and pickling but no hash.
- Arithmetic broadcasts scalars against arrays: `DateTime` plus or minus
  `TimeDelta` gives a `DateTime`, and `DateTime` minus `DateTime` a
  `TimeDelta`, an array one when either operand is an array, as in
  `DateTime("2023-01-01") + TimeDelta([30, 90], "min")`.
- It rejects `NaT` and resolutions of weeks, months, and years.
- An epoch, of `Times` or `with_epoch`, is a scalar `DateTime`; an array
  one raises `TypeError`.
- Array `DateTime` is Python only; C++ stores only the scalar epoch.

## Inputs

- **`Times(times, unit=None, *, epoch=None)`** takes any of these:
  - durations, as a `timedelta64` array, a sequence of `TimeDelta`, or a
    pandas `TimedeltaIndex`;
  - dates, as today;
  - numbers, which need `unit`.

  For durations and dates, `unit` defaults to their resolution, such as
  minutes for a `timedelta64[m]` array, or to the finest unit of a sequence
  of durations, and sets only the storage.
- **`TimeSeries(values, times)`** takes `times` as a `Times`, or durations
  or dates as `Times` accepts them. Plain numbers raise `TypeError`, naming
  `Times(numbers, unit)`.
- **`TimeSeries.from_pandas`** takes `unit` for storage. A numeric index
  needs it.
- **`TimeSeries.evaluate(t)` and `TimeSeriesTensor.evaluate(t)`** take a
  duration or an array of durations, measured from time 0. A series with an
  epoch also takes dates, as today. Numbers raise `TypeError`. The `unit`
  argument is removed.
- **`Window(size, step)` and `TimeDelayEmbedding(delay)`** take durations
  only. Numbers raise `TypeError`.

## Outputs

Instants are dates when the series has an epoch, and durations from time 0
otherwise. Arrays of them come out at the finer of the series' unit and
microseconds. Microseconds keep dates within about 290,000 years of 1970;
finer units narrow that range, to 1678-2262 for nanoseconds.

- **`np.asarray(times)`** gives `datetime64` or `timedelta64` instead of
  floats. `Times.to_floats(unit=None)` gives the float64 timestamps, in
  `unit` or the unit of the times, for code that needs numbers.
  `to_datetime64` stays.
- **`TimeSeries.times`** for several channels is a `TimesTensor` of shape
  `(n_channels,)`, one `Times` per channel in the series' unit, instead of
  a float `NestedTensor`.
- **`TimeSeriesTensor.times`** is a `TimesTensor` of the tensor's shape,
  one `Times` per series, such as the times of every window, with a last
  axis of `n_channels` for series of several channels.
- **`TimeSeries.domain`** has `DateTime` or `TimeDelta` ends. For a series
  with an epoch, these are the instants it can be evaluated at, so a
  window's domain is where it lies.
- **`TimeSeries.origin`** is a `TimeDelta`, or a tuple of them for several
  channels: where time 0 lies in the series a window was cut from.
- **`TimeSeries.bounds`** is removed. With an epoch, `domain` already gives
  where a window lies. Without one, that is `domain` shifted by `origin`.
- **`Window(..., return_bounds=True)`** gives a `datetime64` or
  `timedelta64` array of shape `(n_windows, 2)` instead of a
  `FloatTensor`.
- **`Window.size`, `Window.step`, and `TimeDelayEmbedding.delay`** are
  `TimeDelta`.

## `TimesTensor`

`TimesTensor` is a tensor whose elements are `Times`, with dtype
`sb.times`, as `TimeSeriesTensor` holds `TimeSeries`. Each element keeps
its own timestamps, unit, and epoch.

- It is a C++ `Tensor<sb::Times>`, registered with the other element types
  in both the CPU and CUDA modules; `sb::Times` gains an empty default.
- It has what other tensors of objects have: `shape`, indexing to a
  `Times` or a `TimesTensor`, assignment of a `Times`, `copy`,
  `array_equal`, iteration, and saving and loading.
- It has no NumPy conversion, since its elements differ in length; index
  it to get a `Times`, then convert that.

## Unchanged

- C++ keeps storing and computing times as doubles in a unit; durations
  and dates are converted at the Python boundary.
- `snap_tolerance`, which is relative and has no unit.
- `TimeSeries.unit`, `to(unit)`, and `plot_timeseries(unit=...)`, which
  choose the storage and the plot axis.

## Out of scope

- A C++ duration type.
- Calendar units (months, years) and time zones; dates stay naive UTC.
