# Point clouds from time series: spec

Issue: [#267](https://github.com/kthtda/stablebear/issues/267)

Plan: [timeseries-functions-plan.md](timeseries-functions-plan.md), phase B

## Goal

Persistent homology of a time series works on point clouds sampled from it,
such as the points a delay embedding or its windows trace. Today users
sample by hand, evaluating every window at a grid of times. One call should
give a `PointCloud` or `PointCloudTensor` in a few standard modes.

## Decisions

These settle the open questions of the issue.

1. **Signature.** One function, `to_point_clouds`, whose arguments choose
   the mode: none for every observation, `n` or `step` for a uniform grid,
   `times` for given times.
2. **Scalar observations.** A series of numbers gives one-dimensional point
   clouds, one point of dimension 1 per time.
3. **Channels.** `channels="join"`, the default, gives one point per time
   with the channels' values side by side. `channels="separate"` gives one
   cloud per channel.
4. **Ends of the domain.** "Every observation" takes the times a series
   stores. A window stores the value in effect at its start at time 0, so
   that point is included. Its end adds no point: the value there is that of
   its last observation, or is itself stored if the end is closed.

## Behavior

### `to_point_clouds(series, times=None, *, n=None, step=None, channels="join")`

In `stablebear.timeseries`.

- `series` is a `TimeSeries`, a `TimeSeriesTensor` such as windows, or a
  pandas `Series` or `DataFrame`, converted with `TimeSeries.from_pandas`.
- Observations must be numbers of shape `()` or `(m,)`, giving points of
  dimension 1 or `m`. `float32` gives `pcloud32`; other numbers give
  `pcloud64`. Other element types or shapes raise `TypeError`.
- A `TimeSeries` gives a `PointCloud`, and a `TimeSeriesTensor` of shape
  `S` a `PointCloudTensor` of shape `S`, whose clouds can differ in size.
  With `channels="separate"`, the result gains a last axis of length
  `n_channels`: a `TimeSeries` gives a `PointCloudTensor` of shape
  `(n_channels,)`, and a tensor one of shape `S + (n_channels,)`.
- The points come in the order of their times.

### Modes

- **Every observation** (no `times`, `n`, or `step`): one point per time the
  series stores. With `channels="join"` and several channels, the times are
  those at which any channel stores an observation, within the domain, plus
  its start; each channel gives the value in effect at each time.
- **Uniform grid, `n`**: `n` evenly spaced times over each series' domain,
  in its own time from 0, including each end the domain includes. With both
  ends closed they are `length / (n - 1)` apart, with one end open
  `length / n` apart starting from the closed end, and with both open
  `length / (n + 1)` apart. Windows of one size give clouds of `n` points
  each. `n` is a positive integer; `n = 1` needs a closed end.
- **Uniform grid, `step`**: times `start + k * step` in each series'
  domain, for `k = 0, 1, ...`, leaving out `start` if that end is open.
  `step` is a positive duration, or a number for a series without a unit,
  as for `cut_windows`.
- **Given times** (`times`): the same times for every series, as
  `evaluate` takes them: durations, numbers for a series without a unit, or
  dates for a series with an epoch. A time outside a domain raises
  `ValueError`, as `evaluate` does.

### Errors

- `ValueError` if more than one of `times`, `n`, and `step` is given, or
  for an invalid `n`, `step`, or `channels`.
- `TypeError` for unsupported observations, and for numbers on a series
  with a unit or durations on one without, as `evaluate` raises.

## Out of scope

- Random sampling: users subsample the resulting point clouds.
- Weights or times as an extra coordinate of the points.
