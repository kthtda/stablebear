# Durations and dates instead of numbers for time series: spec

Issue: [#258](https://github.com/kthtda/stablebear/issues/258), which this
grew out of

Plan: [pandas-timeseries-plan.md](pandas-timeseries-plan.md), phases C to G

## Goal

A plain number such as `series(3.5)` or `Window(size=3.0)` means
something only together with the unit of the series, which the reader
cannot see. The time series API should take and give times as durations
and dates instead, so that every time says what it measures. A series
whose times measure no physical time, such as sample indices, has no unit
and keeps plain numbers. Time series are in beta, so the old forms are
removed rather than deprecated.

## Principle

- A **duration** is a `TimeDelta`, `np.timedelta64`, `datetime.timedelta`,
  or `pd.Timedelta`.
- A **date** is a `DateTime`, or a value it accepts.
- A **time** of a series is a duration measured from its time 0. For a
  series with an epoch, it can also be a date. A window's time 0 is its
  start.
- A series without an epoch gives instants, such as its domain, as
  durations, and a series with an epoch gives them as dates. Its own
  timestamps are always durations from time 0, which are exact.
- The `unit` of a series remains, but only as its storage: its times are
  floats in that unit. It sets their precision and the unit of a plot
  axis. No number in the API is read in it.
- A series without a unit takes and gives plain numbers instead (see
  below). Whether a series has a unit decides which of the two it takes,
  so no number is ever read in a unit the reader cannot see.
- There is no `Times` type: durations are `TimeDelta` and
  `TimeDeltaTensor`, and dates are `DateTime` and `DateTimeTensor`.

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
- A `TimeDelta` holds one duration; a `TimeDeltaTensor` holds an array of
  them.
- A `TimeDelta` given to `Window`, `TimeDelayEmbedding`, or `evaluate` is
  converted to the series' unit by an exact factor, without rounding.
  `evaluate` and `TimeSeries` take a `TimeDeltaTensor` as an array of
  durations.
- `TimeDelta` is Python only; nothing in C++ stores a duration.

## `TimeDeltaTensor` and `DateTimeTensor`

`TimeDelta` and `DateTime` hold one value each. A tensor of durations is a
`TimeDeltaTensor`, and a tensor of instants is a `DateTimeTensor`. Each
keeps its counts as one tensor in one unit.

- `TimeDeltaTensor(counts, unit)` takes a `FloatTensor`, `IntTensor`,
  `ndarray`, or nested list of numbers, as in
  `TimeDeltaTensor(sb.FloatTensor([[30, 90], [240, 60]]), "min")`, kept as
  a `FloatTensor`, or an `IntTensor` for integers so that their counts stay
  exact. Without `unit`, it takes a `timedelta64` array or a
  `TimedeltaIndex`, kept at its resolution, or a sequence of durations, in
  their finest unit.
- `DateTimeTensor(values)` takes a `datetime64` array, a pandas
  `DatetimeIndex` or `Series` of dates, or a sequence of values `DateTime`
  accepts, such as strings. The instants are kept at their resolution, or
  the finest of a sequence, as an `IntTensor` of counts after
  1970-01-01T00:00. A timezone is converted to UTC and dropped, as for a
  `DateTime`.
- Both work as their scalars do, elementwise:
  - they have `shape`, `ndim`, `len`, iteration, and indexing, which gives
    a `TimeDelta` or `DateTime` for one element and a tensor otherwise;
  - `count` is the tensor of counts and `unit` its unit;
    `to_timedelta64` and `to_datetime64` give `timedelta64` and
    `datetime64` arrays;
  - `<`, `<=`, `>`, `>=`, `==`, and `!=` give a `BoolTensor`, as tensors
    do; `==` compares count and unit, as for a scalar. `array_equal` tells
    whether two are equal as a whole;
  - they have `repr` and pickling but no hash.
- Arithmetic is that of the scalars, elementwise, with a scalar broadcast
  against a tensor, in the finer unit by exact factors. The result is a
  tensor when either operand is one: `TimeDeltaTensor` plus or minus a
  duration, negation, and multiplication by a number give a
  `TimeDeltaTensor`; a date plus or minus a duration gives a
  `DateTimeTensor`, as in
  `DateTime("2023-01-01") + TimeDeltaTensor([30, 90], "min")`; and a date
  minus a date gives a `TimeDeltaTensor`.
- They reject `NaT`, counts that are not finite, and weeks, months, and
  years.
- An epoch, of `TimeSeries` or `with_epoch`, is a `DateTime`; a
  `DateTimeTensor` raises `TypeError`.
- Both are Python only and have no C++ element type.

## Inputs

- **`TimeSeries(values, times, *, epoch=None)`** takes `times` as any of
  these:
  - durations, as a `TimeDeltaTensor`, a `timedelta64` array, a sequence
    of `TimeDelta`, or a pandas `TimedeltaIndex`, measured from time 0;
  - dates, as a `DateTimeTensor`, a `datetime64` array, or a pandas
    `DatetimeIndex` or `Series` of dates.

  The unit of the series is the resolution of the times, such as minutes
  for a `timedelta64[m]` array, the unit of a `TimeDeltaTensor`, or the
  finest unit of a sequence of durations. `to(unit)` converts the stored
  times, domain ends, origins, and lags by one exact factor each, so the
  unit is always the unit the times are stored in. Plain numbers give a
  series without a unit.
- **`epoch`** is the date that time 0 stands for, a `DateTime` or a value
  it accepts. Dates keep their instants and are counted from it instead of
  from the first of them, in integers and converted to the unit once, so
  whole multiples of the unit are exact. Durations have their time 0 at
  it, as `with_epoch` sets it. `with_epoch` stays, to set, replace, or
  remove the epoch of a series without shifting its times.
- **`TimeSeries.from_channels`** converts every channel into the unit of
  the series, `unit` or the unit the channels share, so all channels of a
  series store their times in its unit. It takes dated channels with
  different epochs and counts them all from the earliest: each channel's times and
  domain end are shifted by the difference of its epoch, converted to its
  unit by an exact factor. Whole multiples of the unit stay exact only up
  to that floating-point addition. Channels with and without an epoch
  still raise.
- **A tuple `(count, unit)`** is a duration wherever one is taken:
  `TimeDelta(count, unit)`, or `TimeDeltaTensor(count, unit)` when `count`
  is a sequence or array, as in `Window(size=(2.0, "h"))` and
  `TimeSeries(values, times=([0.0, 1.5], "s"))`. Only a tuple whose second
  element is a unit is read so; any other tuple keeps its meaning.
- **`TimeSeries.from_pandas`** takes `unit` for storage. A numeric index
  with `unit` is a count of that unit, and without it gives a series
  without a unit.
- **`TimeSeries.evaluate(t, unit=None)` and
  `TimeSeriesTensor.evaluate(t, unit=None)`** take a duration or an array
  of durations, measured from time 0. With `unit`, `t` is a count or an
  array of counts of it, as in `series(90, "min")`, the same as
  `series(TimeDelta(90, "min"))`. A series with an epoch also takes dates,
  including a `DateTime` and a `DateTimeTensor`. On a series with a unit,
  numbers without `unit` raise `TypeError`, naming `series(t, unit)`.
- **`Window(size, step)` and `TimeDelayEmbedding(delay)`** take durations
  for series with a unit, and numbers for series without one; the other
  kind raises `TypeError` when the window or embedding is applied.

## Outputs

Instants are dates when the series has an epoch, and durations from time 0
otherwise. Arrays of them come out at the finer of the series' unit and
microseconds. Microseconds keep dates within about 290,000 years of 1970;
finer units narrow that range, to 1678-2262 for nanoseconds.

- **`TimeSeries.times`** is a `TimeDeltaTensor` of shape
  `(n_observations,)`: the timestamps as float counts in the series'
  unit, from its time 0, with or without an epoch. Their dates are
  `series.epoch + series.origin + series.times`, rounded as a
  `DateTimeTensor` is; the origin places a window in the series it was cut
  from. `count` gives the floats for code that needs numbers.
- **`TimeSeries.datetimes`** is that `DateTimeTensor`, at the finer of the
  series' unit and microseconds. It raises `ValueError` for a series
  without an epoch, naming `with_epoch`, and for several channels, naming
  `series[i].datetimes`.
- **`TimeSeries.times` of several channels** raises `ValueError`, since
  the channels differ in length, naming `series[i].times`.
- **`TimeSeriesTensor.times`** is removed; `windows[i].times` gives the
  times of one window.
- **`TimeSeries.domain`** has `DateTime` or `TimeDelta` ends. For a series
  with an epoch, these are the instants it can be evaluated at, so a
  window's domain is where it lies.
- **`TimeSeries.origin`** is a `TimeDelta`, or a `TimeDeltaTensor` of
  shape `(n_channels,)` for several channels, in the series' unit: where
  time 0 lies in the series a window was cut from.
- **The channels of a window share its origin**, the start `a_k` as
  computed, which is also its bound. A channel whose observation the start
  snaps to still holds that observation at time 0, so its placement in
  the series is off by at most the snap tolerance. Channels combined with
  `TimeSeries.from_channels` can still have different origins.
- **`TimeSeries.bounds`** is removed. With an epoch, `domain` already gives
  where a window lies. Without one, that is `domain` shifted by `origin`.
- **`Window(..., return_bounds=True)`** gives a `TimeDeltaTensor` of
  shape `(n_windows, 2)` in the series' unit, holding the bounds as
  computed, instead of a `FloatTensor`. For a series with an epoch, it
  gives a `DateTimeTensor` of where the windows lie, at the finer of the
  series' unit and microseconds.
- **`Window.size`, `Window.step`, and `TimeDelayEmbedding.delay`** are
  `TimeDelta`, or floats when given as numbers for a series without a
  unit.

## Series without a unit

A series built from plain numbers, as in
`TimeSeries(values, times=[0.0, 1.0, 1.5])`, has no unit: `unit` is
`None`. Its times are numbers, not durations, and it works as a series
with a unit does, with numbers in place of durations:

- `evaluate`, `Window`, and `TimeDelayEmbedding` take numbers, as in
  `series(2.5)` and `Window(size=3.0)`. A duration, a date, a tuple
  `(count, unit)`, or `unit=` on it raises `TypeError`.
- `times` is a `FloatTensor`, `domain` has float ends, `origin` is a float
  or a `FloatTensor`, and `return_bounds` gives a `FloatTensor`.
- It has no epoch: `epoch=` and `with_epoch` raise `ValueError`.
- `to(unit)` gives it a unit: its numbers become counts of `unit`, so
  times `[0.0, 1.5]` become 0 s and 1.5 s with `to("s")`.
- `from_channels` combines channels without a unit, and raises for
  channels with and without one.
- `plot_timeseries` labels its axis "time", with no unit; it cannot share
  an axis with a series that has a unit.
- C++ stores it with a `TimeUnit` that converts only to itself; it is
  saved and loaded as any unit is.

## No `Times`

`Times` and `TimesTensor` are removed from the Python API, with
`sb.times` and the saving and loading of a `TimesTensor`. Their uses move
to the time series:

- building times from durations or dates: `TimeSeries(values, times)`;
- a unit for numbers: `TimeDeltaTensor(numbers, unit)`, or none for a
  series without a unit;
- an epoch: `TimeSeries(..., epoch=...)` for durations, `with_epoch`,
  and `from_channels` for channels of different epochs;
- converting the unit: `TimeSeries.to(unit)`;
- the timestamps as floats or dates: `series.times.count` and
  `series.datetimes`.

C++ keeps `sb::Times` as the storage of a series; `Tensor<sb::Times>` is
no longer registered.

## Unchanged

- C++ keeps storing and computing times as doubles in a unit; durations
  and dates are converted at the Python boundary.
- `snap_tolerance`, which is relative and has no unit.
- `TimeSeries.unit`, `to(unit)`, and `plot_timeseries(unit=...)`, which
  choose the storage and the plot axis.

## Out of scope

- A C++ duration type.
- Calendar units (months, years) and time zones; dates stay naive UTC.
