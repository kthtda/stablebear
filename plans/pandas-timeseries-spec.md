# pandas and NumPy input for time series: spec

Issue: [#258](https://github.com/kthtda/stablebear/issues/258)

Plan: [pandas-timeseries-plan.md](pandas-timeseries-plan.md), phases A, B,
and H to L

The [durations spec](timeseries-durations-spec.md) changes how times are
given and returned; where the two differ, it takes precedence.

## Goal

Most time series reach Python as a pandas `Series` or `DataFrame`. Users
should be able to build a `TimeSeries` from one in one call, window and embed
it without converting first, and get results back as pandas.

## Decisions

These settle the open questions of the issue.

1. **Time zones.** Timezone-aware input is converted to UTC and the zone is
   dropped. This applies to `DateTime`, `DateTimeTensor`, `TimeSeries`,
   calling a series, and `from_pandas`, for both `datetime` and pandas values.
   Nothing stores a zone, so `to_pandas` returns naive UTC dates.
2. **Missing data.** `NaN` and `pd.NA` values are kept as observations, as
   `TimeSeries(values, times)` keeps them today. `NaT` in an index is
   rejected. Users call `dropna()` first if they want to drop rows.
3. **DataFrame shape.** A `DataFrame` becomes one channel per column. All
   channels share the times of the index. A vector-valued channel remains
   one call: `TimeSeries(df.to_numpy(), df.index)`.
4. **Index order.** An unsorted index or one with duplicates is rejected,
   as `TimeSeries` rejects such timestamps. The error names `sort_index()`
   or `index.duplicated()`.
5. **Resolution.** The default `unit`, which sets only how the times are
   stored, is the resolution of a `DatetimeIndex` or `TimedeltaIndex`,
   mostly microseconds in pandas 3 and nanoseconds in pandas 2, as for
   `datetime64` and `timedelta64` arrays.
6. **Return types.** `cut_windows` and `delay_embed` return the same
   stablebear types for pandas input as for a `TimeSeries`, so results feed
   straight into `compute_persistent_homology`. Calling a series at dates
   returns tensors as now.
7. **Bare NumPy arrays.** `cut_windows` and `delay_embed` accept a
   `TimeSeries` or a pandas `Series` or `DataFrame`. NumPy data goes through
   `TimeSeries(values, times)`, which is already one call.

## Behavior

### pandas values in existing entry points

- `DateTime(pd.Timestamp(...))` keeps the timestamp's resolution, such as
  nanoseconds. Today `np.datetime64(timestamp)` drops it to microseconds.
- Durations (the `size` and `step` of `cut_windows`, the `delay` of `delay_embed`)
  accept `pd.Timedelta` at its own resolution. It is a `datetime.timedelta`
  and is accepted today, but rounded to microseconds.
- `DateTime` converts a timezone-aware `datetime` or `pd.Timestamp` to the
  same instant in naive UTC instead of raising.
- The `times` of `TimeSeries(...)` accept a `DatetimeIndex` or a `Series`
  of dates, naive or timezone-aware. A naive index works today; an aware
  one fails with an unclear `TypeError`.
- Calling a `TimeSeries` accepts a `DatetimeIndex` or a `Series` of dates as
  it accepts a `datetime64` array. Today an index is taken as numbers.

### `TimeSeries.from_pandas(obj, unit=None, *, snap_tolerance=1e-14)`

- A `Series` gives one channel; a `DataFrame` gives one channel per column,
  in column order. Each column is converted as a `Series` is, so `pd.NA`
  becomes `NaN`, and the columns then take one common dtype, since all
  channels of a series have one element type.
- The index gives the times:
  - `DatetimeIndex`: counted from the first timestamp, which becomes the
    epoch, and converted to `unit`, as for `datetime64` arrays.
  - `TimedeltaIndex`: converted to `unit`, with no epoch.
  - Numeric index, including `RangeIndex`: counts of `unit`, with no
    epoch, or without `unit` a series without a unit (see the
    [durations spec](timeseries-durations-spec.md)).

  There is no `epoch` keyword; `with_epoch` sets or changes the epoch.
- Raises `TypeError` for other objects, other index types (such as
  `PeriodIndex` and `MultiIndex`), and values that are not numeric or
  boolean. Raises `ValueError` for `NaT` in the index, an unsorted index, a
  duplicated timestamp, and a `DataFrame` with no columns.
- Column names and the series name are not kept; a `TimeSeries` has no
  channel names.

### `TimeSeries.to_pandas(resolution=None)`

- The index of a series with an epoch is a `DatetimeIndex` of
  `epoch + origin + times`, else a `TimedeltaIndex` of `times` (see the
  [durations spec](timeseries-durations-spec.md)). `resolution` is one of
  `"s"`, `"ms"`, `"us"`, and `"ns"`, the resolutions pandas supports. It
  defaults to the finer of the series' unit and microseconds. A unit finer
  than nanoseconds raises unless `resolution="ns"` is passed, so that
  rounding to nanoseconds, which can merge times, is chosen explicitly.
- `TimeSeries.to_pandas()` returns, for one channel, a `Series` with that
  index. For several channels it returns a `DataFrame` with columns `0` to
  `n_channels - 1`, indexed by the union of the channels' times. Each
  column holds its channel's value in effect at each time, as calling the
  series gives it, and `NaN` outside the channel's domain, so a column keeps its
  integer or boolean dtype when every time lies in its domain.
- Only observations of shape `()` that are numbers or booleans can be
  converted; other series raise `TypeError`.
- If pandas is not installed, it raises `ImportError` saying that the
  pandas package is needed, without naming an installer.

### `cut_windows` and `delay_embed` on pandas input

`cut_windows(obj, ...)` and `delay_embed(obj, ...)` accept a pandas
`Series` or `DataFrame`, convert it with `TimeSeries.from_pandas(obj)`, and
return what they return for that series. Other objects that are not a
`TimeSeries` raise `TypeError`, naming `TimeSeries(values, times)` and
`TimeSeries.from_pandas`.

### Dependency

pandas stays optional, and stablebear declares no extra for it: users
install it themselves. The `test` dependency group and
`docs/requirements.txt` include it. stablebear never imports pandas at
module level. Code that receives a pandas object uses it through that
object's module, so only `to_pandas` needs to import it.

## Out of scope

- Storing a time zone, or channel names.
- Accepting `(values, times)` tuples or bare arrays in `cut_windows` and
  `delay_embed`.
- Returning pandas objects from `cut_windows`, `delay_embed`, or calling
  a series.
