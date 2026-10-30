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
   dropped. This applies to `DateTime`, `Times`, `evaluate`, and
   `from_pandas`, for both `datetime` and pandas values. Nothing stores a
   zone, so `to_pandas` returns naive UTC dates.
2. **Missing data.** `NaN` and `pd.NA` values are kept as observations, as
   `TimeSeries(values, times)` keeps them today. `NaT` in an index is
   rejected. Users call `dropna()` first if they want to drop rows.
3. **DataFrame shape.** A `DataFrame` becomes one channel per column. All
   channels share the times of the index. A vector-valued channel remains
   one call: `TimeSeries(df.to_numpy(), df.index)`.
4. **Index order.** An unsorted index or one with duplicates is rejected,
   as `Times` rejects such timestamps. The error names `sort_index()` or
   `index.duplicated()`.
5. **Resolution.** The default `unit`, which sets only how the times are
   stored, is the resolution of a `DatetimeIndex` or `TimedeltaIndex`,
   usually nanoseconds, as for `datetime64` and `timedelta64` arrays.
6. **Return types.** `Window` and `TimeDelayEmbedding` return the same
   stablebear types for pandas input as for a `TimeSeries`, so results feed
   straight into `compute_persistent_homology`. `evaluate` at dates returns
   tensors as now.
7. **Bare NumPy arrays.** `Window` and `TimeDelayEmbedding` accept a
   `TimeSeries` or a pandas `Series` or `DataFrame`. NumPy data goes through
   `TimeSeries(values, times)`, which is already one call.
8. **NumPy protocols.** Tensors keep refusing NumPy ufuncs, but with an
   error that says what to do instead. `__array_function__` and
   `__dlpack__` are out of scope.

## Behavior

### pandas values in existing entry points

- `DateTime(pd.Timestamp(...))` keeps the timestamp's resolution, such as
  nanoseconds. Today `np.datetime64(timestamp)` drops it to microseconds.
- Durations (`Window` `size` and `step`, `TimeDelayEmbedding` `delay`)
  accept `pd.Timedelta` at its own resolution. It is a `datetime.timedelta`
  and is accepted today, but rounded to microseconds.
- `DateTime` converts a timezone-aware `datetime` or `pd.Timestamp` to the
  same instant in naive UTC instead of raising.
- `Times(...)` and the `times` of `TimeSeries(...)` accept a
  `DatetimeIndex` or a `Series` of dates, naive or timezone-aware. A naive
  index works today; an aware one fails with an unclear `TypeError`.
- `TimeSeries.evaluate` accepts a `DatetimeIndex` or a `Series` of dates as
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
  - Numeric index, including `RangeIndex`: numbers in `unit`, with no
    epoch.

  There is no `epoch` keyword; `with_epoch` sets or changes the epoch.
- Raises `TypeError` for other objects, other index types (such as
  `PeriodIndex` and `MultiIndex`), and values that are not numeric or
  boolean. Raises `ValueError` for `NaT` in the index, an unsorted index, a
  duplicated timestamp, and a `DataFrame` with no columns.
- Column names and the series name are not kept; a `TimeSeries` has no
  channel names.

### `Times.to_pandas(resolution=None)` and `TimeSeries.to_pandas(resolution=None)`

- `Times.to_pandas()` returns a `DatetimeIndex` when the times have an
  epoch, else a `TimedeltaIndex`, as `np.asarray(times)` gives them (see
  the [durations spec](timeseries-durations-spec.md)). `resolution` is one
  of `"s"`, `"ms"`, `"us"`, and `"ns"`, the resolutions pandas supports. It
  defaults to the resolution of `np.asarray(times)`, or nanoseconds if that
  is finer.
- `TimeSeries.to_pandas()` returns, for one channel, a `Series` indexed by
  its `times.to_pandas()`. For several channels it returns a `DataFrame`
  with columns `0` to `n_channels - 1`, indexed by the union of the
  channels' times, with `NaN` where a channel has no observation at a time.
- Only observations of shape `()` that are numbers or booleans can be
  converted; other series raise `TypeError`.
- If pandas is not installed, both raise `ImportError` naming
  `pip install pandas`.

### `Window` and `TimeDelayEmbedding` on pandas input

`Window(...)(obj)` and `TimeDelayEmbedding(...)(obj)` accept a pandas
`Series` or `DataFrame`, convert it with `TimeSeries.from_pandas(obj)`, and
return what they return for that series. Other objects that are not a
`TimeSeries` raise `TypeError`, naming `TimeSeries(values, times)` and
`TimeSeries.from_pandas`.

### NumPy ufuncs on tensors

`ndarray + tensor` and the other operators with an array on the left keep
working, as they do today because NumPy defers to the tensor's reflected
operator. Other ufuncs, such as `np.sin(tensor)`, raise `TypeError`. For a
numeric tensor the message suggests `np.asarray(tensor)`; for other tensors
it says they have no NumPy equivalent.

### Dependency

pandas stays optional, and stablebear declares no extra for it: users
install it themselves. The `test` dependency group and
`docs/requirements.txt` include it. stablebear never imports pandas at
module level. Code that receives a pandas object uses it through that
object's module, so only `to_pandas` needs to import it.

## Out of scope

- Storing a time zone, or channel names.
- Accepting `(values, times)` tuples or bare arrays in `Window` and
  `TimeDelayEmbedding`.
- Returning pandas objects from `Window`, `TimeDelayEmbedding`, or
  `evaluate`.
- Dispatching NumPy ufuncs to stablebear operations, `__array_function__`,
  and DLPack. These need their own issue.
