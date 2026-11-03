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
3. **Channels.** `TimeSeries.stack_channels` combines the channels of a
   series into one, as `sb.stack` stacks tensors, so `to_point_clouds`
   takes series of one channel only. One cloud per channel is
   `to_point_clouds(series[i])`.
4. **Ends of the domain.** "Every observation" takes the times a series
   stores. A window stores the value in effect at its start at time 0, so
   that point is included. Its end adds no point: the value there is that of
   its last observation, or is itself stored if the end is closed.

## Behavior

### `TimeSeries.stack_channels(axis=0)`

- Returns a series of one channel whose observation at each time stacks
  the channels' observations in effect there along a new axis `axis` of
  length `n_channels`, as `sb.stack` stacks tensors: shape `(n_channels,
  *shape)` for `axis=0`. Any element type works. A negative `axis` counts
  from the end.
- The times are those at which any channel stores an observation within
  the domain, and the start of the domain. The domain stays the same, with
  its ends open or closed as before.
- Observations can be dropped or repeated, and the docstring warns of it:
  a channel's observations outside the domain, or at an end the domain
  leaves out, are dropped, and an observation in effect at several of the
  times is repeated at each.
- The unit and epoch stay the same. The snap tolerance is the smallest of
  the channels'.
- Raises `ValueError` if the channels have different shapes, as `sb.stack`
  does, or different origins, if `axis` is out of range, or if a channel has
  no observations.

### `to_point_clouds(series, times=None, *, n=None, step=None)`

In `stablebear.timeseries`.

- `series` is a `TimeSeries`, a `TimeSeriesTensor` such as windows, or a
  pandas `Series` or `DataFrame`, converted with `TimeSeries.from_pandas`.
- Observations must be numbers of shape `()` or `(m,)`, giving points of
  dimension 1 or `m`. `float32` gives `pcloud32`; other numbers give
  `pcloud64`. Other element types or shapes raise `TypeError`.
- A `TimeSeries` gives a `PointCloud`, and a `TimeSeriesTensor` of shape
  `S` a `PointCloudTensor` of shape `S`, whose clouds can differ in size.
- A series of several channels raises `ValueError`, naming
  `stack_channels` and `series[i]`.
- The points come in the order of their times.

### Modes

- **Every observation** (no `times`, `n`, or `step`): one point per time the
  series stores.
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

- `ValueError` if more than one of `times`, `n`, and `step` is given, for
  an invalid `n` or `step`, or for several channels.
- `TypeError` for unsupported observations, and for numbers on a series
  with a unit or durations on one without, as `evaluate` raises.

## Out of scope

- Random sampling: users subsample the resulting point clouds.
- Weights or times as an extra coordinate of the points.
